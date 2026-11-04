"""Semantic counting with original-colour Fluent artwork and the base scenes' placement.

Styles and image transforms never change the counted class. All scene assets
come from the requested split, including distractors. Occlusion is bounded to
keep category cues readable; overlap-relation questions have their own module.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from PIL import Image

from shared.iconlib.fluent import ALPHA_THRESHOLD

from .base import scenes
from .difficulty import COUNT_RANGES, TARGET_RANGES


STYLE_ORDER = ("3d", "color", "flat", "high_contrast")
AREA_BUDGET_FRACTIONS = (0.18, 0.32)
MIN_LONGEST_AT_448 = 28.0
MAX_LONGEST_FRACTION = 0.36
MIN_AREA_AT_448 = 250.0
DISTRACTOR_CLASS_COUNTS = {
    "easy": 2, "medium": 3, "hard": 4,
    "xhard": 6, "xxhard": 8, "ood": 10,
}
ANGLE_LIMITS = {"easy": 15, "medium": 25, "hard": 40,
                "xhard": 60, "xxhard": 90}

# These scene-only exclusions supplement the inventory's family review. They
# cover broad nouns and distinctions that disappear in High Contrast; they do
# not merge classes, remove targets, or change train/eval membership.
SCENE_EXCLUSION_GROUPS = {
    "oval_footballs": {"american_football", "rugby_football"},
    "stitched_balls": {"baseball", "softball"},
    "coats": {"coat", "lab_coat"},
    "drums": {"drum", "long_drum", "oil_drum"},
    "eyewear": {"glasses", "sunglasses", "goggles", "diving_mask"},
    "pens": {"pen", "fountain_pen", "black_nib"},
    "telephones": {"telephone", "telephone_receiver", "mobile_phone"},
    "mice": {"mouse", "computer_mouse", "trackball"},
    "seats": {"chair", "seat", "manual_wheelchair", "motorized_wheelchair"},
    "bells": {"bell", "bellhop_bell"},
    "keyboards": {"keyboard", "musical_keyboard"},
    "cable_cars": {"aerial_tramway", "mountain_cableway"},
    "cups_with_straws": {"cup_with_straw", "bubble_tea"},
    "hot_drink_cups": {"hot_beverage", "teacup_without_handle", "mate"},
    "drinking_glasses": {"cocktail_glass", "tropical_drink", "tumbler_glass",
                         "wine_glass", "glass_of_milk"},
}


def _scene_exclusion_groups(class_id):
    return sorted(name for name, members in SCENE_EXCLUSION_GROUPS.items()
                  if class_id in members)


def _class_records(lib, split):
    if split not in ("train", "eval", "val", "test"):
        raise ValueError("Fluent scenes require an explicit train/eval/val/test split")
    classes = {}
    for entry in lib.entries:
        if entry["split"] != split or not entry.get("countable_unit", False):
            continue
        if entry["class_id"] not in classes:
            classes[entry["class_id"]] = {
                "noun": entry["noun"], "plural": entry["plural"],
                "families": set(), "counting_notes": [],
            }
        record = classes[entry["class_id"]]
        record["families"].update(entry.get("family_ids", []))
        record["families"].update(
            f"scene:{name}" for name in _scene_exclusion_groups(entry["class_id"]))
        record["counting_notes"].extend(entry.get("counting_notes", []))
    if not classes:
        raise ValueError(f"no countable Fluent classes in split {split!r}")
    for class_id in classes:
        available = {asset["style"] for asset in lib.variants(class_id, split=split)}
        if set(STYLE_ORDER) != available:
            raise ValueError(f"{class_id!r} does not have all four Fluent styles")
    return classes


def eligible_distractors(lib, split, target_class):
    """Block family siblings and all accessory-bearing negatives conservatively.

    This is a per-scene rule, not a change to the 300/166 class assignment.
    Every class remains eligible as a category target, including a fishing pole
    whose fish accessory must not become a negative example for a fish query.
    """
    records = _class_records(lib, split)
    if target_class not in records:
        raise ValueError(f"target {target_class!r} is not in split {split!r}")
    families = records[target_class]["families"]
    return sorted(class_id for class_id, record in records.items()
                  if class_id != target_class and not record["counting_notes"]
                  and not families.intersection(record["families"]))


def _style_schedule(rng, class_schedule):
    """Guarantee all styles for any class with at least four scene copies."""
    styles_by_class = {}
    for class_id, count in Counter(class_schedule).items():
        cycle = list(STYLE_ORDER)
        rng.shuffle(cycle)
        styles = [cycle[index % len(cycle)] for index in range(count)]
        rng.shuffle(styles)
        styles_by_class[class_id] = styles
    return [styles_by_class[class_id].pop() for class_id in class_schedule]


def _scene_schedule(rng, records, lib, split, query, n, target_class, difficulty,
                    target_n=None):
    class_count = DISTRACTOR_CLASS_COUNTS[difficulty]
    if query == "category":
        if target_class is None:
            target_class = rng.choice(sorted(records))
        if target_class not in records:
            raise ValueError(f"target {target_class!r} is not in split {split!r}")
        candidates = eligible_distractors(lib, split, target_class)
        if not candidates:
            raise ValueError(f"no unambiguous distractors for {target_class!r}")
        target_lo, target_hi = TARGET_RANGES[difficulty]
        target_hi = max(target_lo, min(target_hi, n - class_count))
        if target_n is None:
            target_n = rng.randint(target_lo, target_hi)
        elif not target_lo <= target_n <= target_hi:
            raise ValueError("fixed category target count is outside its difficulty range")
        classes = rng.sample(candidates, min(len(candidates), class_count,
                                              n - target_n))
        schedule = [target_class] * target_n
        schedule.extend(classes)
        schedule.extend(rng.choice(classes) for _ in range(n - len(schedule)))
    else:
        # A detached accessory makes a generic total count ambiguous.
        candidates = sorted(key for key, value in records.items()
                            if not value["counting_notes"])
        if not candidates:
            raise ValueError("no accessory-free Fluent classes for generic counting")
        classes = rng.sample(candidates, min(len(candidates), class_count, n))
        schedule = classes + [rng.choice(classes) for _ in range(n - len(classes))]
    rng.shuffle(schedule)
    return schedule, target_class


def _bounded_area_allocation(weights, lower, upper, budget):
    """Allocate an N-independent scene budget without dropping counting units."""
    weights = np.asarray(weights, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if np.any(lower > upper) or not float(lower.sum()) <= budget <= float(upper.sum()):
        raise ValueError("Fluent area budget is infeasible under readable size bounds")
    low, high = 0.0, float(np.max(upper / weights))
    for _ in range(60):
        middle = (low + high) / 2
        if float(np.clip(weights * middle, lower, upper).sum()) < budget:
            low = middle
        else:
            high = middle
    return np.clip(weights * ((low + high) / 2), lower, upper)


def _sprite_metrics(sprite):
    mask = np.asarray(sprite.getchannel("A")) > ALPHA_THRESHOLD
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Fluent sprite has no foreground after transformation")
    return int(mask.sum()), max(int(xs.max() - xs.min() + 1),
                               int(ys.max() - ys.min() + 1))


def _prepare_sprites(rng, lib, split, schedule, styles, size, difficulty,
                     target_class, area_budget):
    """Draw broad exchangeable area weights, then solve to the scene budget.

    Each asset's own alpha mask supplies the area-to-size conversion. The
    weights never inspect target flags or class labels. Bounded allocation
    preserves tiny-object readability and does not shrink dense scenes after
    placement failures.
    """
    prepared, references = [], []
    minimum_longest = math.ceil(size / 448 * MIN_LONGEST_AT_448)
    maximum_longest = round(size * MAX_LONGEST_FRACTION)
    minimum_area = math.ceil(max(64.0, MIN_AREA_AT_448 * (size / 448) ** 2))
    # Jittered quantiles guarantee broad coverage within every scene; random
    # permutation makes target and distractor assignments exchangeable.
    spread = rng.uniform(1.65, 2.25)
    weights = [math.exp(-spread + 2 * spread * (index + rng.random()) / len(schedule))
               for index in range(len(schedule))]
    rng.shuffle(weights)
    lower, upper = [], []
    for index, (class_id, style) in enumerate(zip(schedule, styles)):
        asset = lib.pick(rng, class_id=class_id, styles=style, split=split)
        if difficulty == "ood":
            # This upside-down orientation interval is absent from every
            # training tier; it supplements the held-out object-count regime.
            angle = rng.choice((-1, 1)) * rng.uniform(100, 180)
        else:
            angle_limit = ANGLE_LIMITS[difficulty]
            angle = rng.uniform(-angle_limit, angle_limit)
        mirror = rng.random() < 0.5
        foreground_size = size * 0.12
        reference = lib.render(asset, size=foreground_size, angle=angle, mirror=mirror)
        reference_area, reference_longest = _sprite_metrics(reference)
        # A small raster margin keeps the final rounded rendering above the
        # declared lower bounds. The upper margin prevents rounding above max.
        lower.append(max(minimum_area * 1.04,
                         reference_area * ((minimum_longest + 1) / reference_longest) ** 2))
        upper.append(reference_area * ((maximum_longest - 1) / reference_longest) ** 2)
        references.append((asset, angle, foreground_size, reference_area, mirror))
    allocations = _bounded_area_allocation(weights, lower, upper, area_budget)
    for index, ((class_id, style), reference, allocation) in enumerate(
            zip(zip(schedule, styles), references, allocations)):
        asset, angle, foreground_size, reference_area, mirror = reference
        scale = math.sqrt(float(allocation) / reference_area)
        best = None
        for _ in range(6):
            sprite = lib.render(asset, size=foreground_size, angle=angle, scale=scale,
                                mirror=mirror)
            sprite = sprite.crop(sprite.getchannel("A").getbbox())
            actual_area, actual_longest = _sprite_metrics(sprite)
            error = abs(actual_area / allocation - 1)
            if (actual_area >= minimum_area
                    and minimum_longest <= actual_longest <= maximum_longest
                    and (best is None or error < best[0])):
                best = (error, sprite, scale, actual_area, actual_longest)
            if actual_area < minimum_area or actual_longest < minimum_longest:
                scale *= 1.012 * max(math.sqrt(minimum_area / actual_area),
                                    minimum_longest / actual_longest)
            elif actual_longest > maximum_longest:
                scale *= 0.99 * maximum_longest / actual_longest
            elif abs(actual_area / allocation - 1) > 0.018:
                scale *= math.sqrt(float(allocation) / actual_area)
            else:
                break
        # Quantization is measured explicitly rather than inferred from scale.
        if best is None:
            raise ValueError("Fluent raster could not satisfy readable footprint bounds")
        _, sprite, scale, actual_area, actual_longest = best
        info = {
            "instance_id": index, "kind": "fluent", "class_id": class_id,
            "icon": asset["identity_id"],
            "identity_id": asset["identity_id"], "asset_id": asset["asset_id"],
            "source_name": asset["source_name"], "noun": asset["noun"],
            "plural": asset["plural"], "style": style, "split": asset["split"],
            "path": asset["path"], "sha256": asset["sha256"],
            "family_ids": list(asset.get("family_ids", [])),
            "scene_exclusion_groups": _scene_exclusion_groups(class_id),
            "counting_notes": list(asset.get("counting_notes", [])),
            "target": class_id == target_class,
            "size_area_weight": weights[index],
            "intended_foreground_pixels": float(allocation),
            "actual_foreground_pixels": actual_area,
            "effective_longest_side": actual_longest,
            "transform": {"angle_degrees": angle, "scale": scale,
                          "foreground_size": foreground_size, "mirror": mirror,
                          "pose_id": asset.get("pose_id")},
        }
        prepared.append((sprite, info))
    diagnostics = {
        "policy": "independent_scene_area_budget",
        "requested_foreground_pixels": area_budget,
        "requested_foreground_fraction": area_budget / (size * size),
        "budget_fraction_range": list(AREA_BUDGET_FRACTIONS),
        "weight_distribution": "shuffled jittered log-uniform quantiles",
        "weight_log_half_range": spread,
        "weight_assignment": "exchangeable across targets and distractors",
        "minimum_foreground_pixels": minimum_area,
        "minimum_longest_side": minimum_longest,
        "maximum_longest_side": maximum_longest,
        "feasible_foreground_pixel_bounds": [float(sum(lower)), float(sum(upper))],
    }
    return prepared, diagnostics


def _place(rng, size, prepared, difficulty):
    extra = {}
    if difficulty == "ood":
        # A held-out annular center prior changes layout without grouping by
        # target flag, class, artwork style, or object area.
        phase = rng.uniform(0, 2 * math.pi)
        anchors = [(size * (0.5 + 0.32 * math.cos(phase + i * math.pi / 4)),
                    size * (0.5 + 0.32 * math.sin(phase + i * math.pi / 4)))
                   for i in range(8)]
        cluster_schedule = [i % len(anchors) for i in range(len(prepared))]
        rng.shuffle(cluster_schedule)
        extra = {"cluster_centers": anchors, "cluster_schedule": cluster_schedule,
                 "cluster_spread_frac": 0.105}
    # Retry the placement without silently lowering N or dropping a target.
    for _ in range(20):
        placed, layer = scenes._scatter(
            rng, size, len(prepared), lambda _rng, slot: prepared[slot],
            ss=1, max_overlap=0.15, min_cdist_frac=0.065,
            same_attribute_overlap_cap=0.08,
            max_overlap_neighbors=2, min_visible_fraction=0.85,
            **extra,
        )
        if len(placed) != len(prepared):
            continue
        return placed, layer
    raise RuntimeError("could not place every Fluent instance unambiguously")


def make_fluent_problem(rng, size, difficulty, split, lib, *, query="category",
                        target_class=None):
    """Return RGB image, question, answer string, CoT, metadata, solutions.

    ``query`` is total or category. Each instance records the 3x3 section of its
    raster foreground centroid, in the base scenes' row-major convention.
    ``eval`` stays an undivided held-out pool; it is never aliased to train.
    """
    if difficulty not in COUNT_RANGES:
        raise ValueError(f"unknown Fluent difficulty: {difficulty!r}")
    if query not in ("total", "category"):
        raise ValueError(f"unknown Fluent query: {query!r}")
    if size < 224:
        raise ValueError("Fluent counting requires size >= 224")
    if target_class is not None and query != "category":
        raise ValueError("target_class only applies to category questions")
    records = _class_records(lib, split)
    # This random scene-level budget is sampled before, and independently of N.
    area_budget = rng.uniform(*AREA_BUDGET_FRACTIONS) * size * size
    n = rng.randint(*COUNT_RANGES[difficulty])
    schedule, target_class = _scene_schedule(
        rng, records, lib, split, query, n, target_class, difficulty,
    )
    target_n = schedule.count(target_class) if query == "category" else None
    for asset_attempt in range(12):
        styles = _style_schedule(rng, schedule)
        try:
            prepared, size_diagnostics = _prepare_sprites(
                rng, lib, split, schedule, styles, size, difficulty, target_class, area_budget,
            )
            break
        except ValueError as error:
            if not ("area budget is infeasible" in str(error)
                    or "raster could not satisfy" in str(error)):
                raise
            if asset_attempt == 11:
                raise RuntimeError("could not satisfy the fixed Fluent scene budget") from error
            # Very thin icons can make a scene's upper area bound too small.
            # Resample appearances and negatives, retaining the answer, total
            # count and previously sampled budget instead of rejecting those.
            schedule, target_class = _scene_schedule(
                rng, records, lib, split, query, n, target_class, difficulty,
                target_n=target_n,
            )
    size_diagnostics["asset_sampling_attempts"] = asset_attempt + 1
    size_diagnostics["asset_retry_keeps"] = ["total_count", "target_class",
                                              "target_count", "scene_budget"]
    placed, layer = _place(rng, size, prepared, difficulty)
    image = Image.new("RGBA", (size, size), "white")
    image.alpha_composite(layer)
    instances = []
    for item in placed:
        section = scenes.section_of(item["cx"], item["cy"], size)
        instances.append({
            **item["info"], "x": item["x"], "y": item["y"],
            "cx": float(item["cx"]), "cy": float(item["cy"]),
            "bbox": [item["x"], item["y"], item["x"] + item["mask"].shape[1],
                     item["y"] + item["mask"].shape[0]],
            "section": section, "foreground_pixels": item["area"],
            "visible_foreground_pixels": item["visible_area"],
            "visible_fraction": item["visible_area"] / item["area"],
        })
    areas = [item["foreground_pixels"] for item in instances]
    longest_sides = [item["effective_longest_side"] for item in instances]
    total_area = sum(areas)
    size_diagnostics.update({
        "actual_foreground_pixels_sum": total_area,
        "actual_foreground_fraction_sum": total_area / (size * size),
        "budget_relative_error": total_area / area_budget - 1,
        "visible_foreground_pixels_union": sum(item["visible_foreground_pixels"]
                                               for item in instances),
        "image_nonwhite_pixels": int(np.any(np.asarray(image.convert("RGB")) < 245,
                                            axis=2).sum()),
        "foreground_area_min": min(areas), "foreground_area_max": max(areas),
        "foreground_area_ratio": max(areas) / min(areas),
        "longest_side_min": min(longest_sides), "longest_side_max": max(longest_sides),
        "longest_side_ratio": max(longest_sides) / min(longest_sides),
        "placement_rescaled": False,
    })
    if query == "category":
        target = records[target_class]
        question = (f"How many {target['plural']} are shown? Count them across all "
                    "drawing styles, sizes, rotations, and mirror images.")
        if target["counting_notes"]:
            question += f" Count each {target['noun']} once; ignore accompanying accessories."
        subset = [item for item in instances if item["class_id"] == target_class]
        cot = f"Count only the {target['plural']} across all drawing styles: {len(subset)}."
    else:
        subset = instances
        question = "How many object icons are shown in total? Count each icon once."
        cot = f"Count every object icon once, regardless of its style or class: {len(subset)}."
    question += " (Answer with a single number.)"
    answer = str(len(subset))
    selected_ids = {item["instance_id"] for item in subset}
    for item in instances:
        item["counted"] = item["instance_id"] in selected_ids
    elements = [{"cx": item["cx"], "cy": item["cy"],
                 "instance_id": item["instance_id"], "class_id": item["class_id"],
                 "info": dict(item)} for item in subset]
    meta = {
        "family": "fluent", "renderer": "fluent_rgba", "mode": query,
        "query": query, "difficulty": difficulty, "split": split, "n": n,
        "count": len(instances), "answer": len(subset), "target_class": target_class,
        "difficulty_parameters": {
            "object_count_range": list(COUNT_RANGES[difficulty]),
            "category_target_range": list(TARGET_RANGES[difficulty]),
            "requested_distractor_classes": DISTRACTOR_CLASS_COUNTS[difficulty],
            "actual_class_count": len(set(schedule)),
            "orientation_degrees": ([[-180, -100], [100, 180]] if difficulty == "ood"
                                    else [-ANGLE_LIMITS[difficulty], ANGLE_LIMITS[difficulty]]),
            "mirror_probability": 0.5,
            "mirror_axis": "horizontal in source coordinates, before rotation",
        },
        "instances": instances,
        "styles": sorted({item["style"] for item in instances}),
        "source": dict(lib.source), "alpha_threshold": ALPHA_THRESHOLD,
        "size_distribution": size_diagnostics,
        "placement": {"implementation": "scenes._scatter", "max_overlap": 0.15,
                      "layout": "annular_cluster_prior" if difficulty == "ood" else "uniform_scatter",
                      "cluster_spread_fraction": 0.105 if difficulty == "ood" else None,
                      "same_identity_overlap_cap": 0.08,
                      "max_overlap_neighbors": 2,
                      "min_visible_fraction": 0.85, "ss": 1},
        "semantic_policy": {
            "category_distractors": "exclude catalog and scene families, and counting_notes entries",
            "scene_exclusion_policy": "fluent_scenes",
            "total_region": "exclude counting_notes entries to avoid accessory counts",
            "identity_and_split": "all styles and transforms retain their class and split",
        },
        "ood_axes": (["heldout_count_range", "upside_down_orientation_interval",
                      "annular_cluster_prior"] if difficulty == "ood" else []),
    }
    return image.convert("RGB"), question, answer, cot, meta, elements
