"""Diverse foreground sizes around the pinned base scene generators.

The scene area budget is drawn before object count or any answer is known.
Per-object foreground areas share that budget using broad random weights.
Only rendering/placement hooks are replaced, in a private function namespace;
the base scenes module and its question/answer rules stay unchanged.
"""
from __future__ import annotations

import collections
import math
import random
import types

import numpy as np
from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

from .base import scenes
from .difficulty import COUNT_RANGES, TARGET_RANGES

POLICY = "independent_scene_ink_budget_six_tiers"
GEOMETRY_BUDGET = dict.fromkeys(("easy", "medium", "hard", "xhard", "xxhard", "ood"), (.18, .30))
LINE_BUDGET = dict.fromkeys(GEOMETRY_BUDGET, (.14, .24))
PAIR_CAP = dict(zip(GEOMETRY_BUDGET, (.06, .10, .16, .22, .27, .30)))
VISIBLE_FLOOR = dict(zip(GEOMETRY_BUDGET, (.94, .90, .84, .78, .73, .70)))
REGION_COUNT = dict(zip(GEOMETRY_BUDGET, (1, 1, 1, 2, 3, 4)))


def _private_scenes():
    """Clone the base scenes' functions so hook writes cannot affect another caller."""
    original = vars(scenes)
    namespace = original.copy()
    for name, value in original.items():
        if isinstance(value, types.FunctionType) and value.__globals__ is original:
            clone = types.FunctionType(value.__code__, namespace, value.__name__,
                                       value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__annotations__ = value.__annotations__
            namespace[name] = clone
    return namespace


def _weights(rng, n):
    # Stratification guarantees both small and large instances when n permits;
    # shuffle before matching weights to shapes/colors/stripe labels.
    values = [math.exp(math.log(32.0) * ((i + rng.random()) / n)) for i in range(n)]
    rng.shuffle(values)
    return np.asarray(values, dtype=float)


def _allocate(total, weights, lower, upper):
    """Water-fill weighted areas with readability and canvas-fit bounds."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    total = float(np.clip(total, lower.sum(), upper.sum()))
    lo, hi = 0.0, float(np.max(upper / weights))
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.clip(mid * weights, lower, upper).sum() < total:
            lo = mid
        else:
            hi = mid
    return np.clip((lo + hi) / 2 * weights, lower, upper), total


def _area_stats(areas):
    values = np.asarray(areas, float)
    return {
        "foreground_pixels": [round(float(x), 2) for x in values],
        "minimum_foreground_pixels": round(float(values.min()), 2),
        "maximum_foreground_pixels": round(float(values.max()), 2),
        "foreground_area_ratio": round(float(values.max() / values.min()), 4),
        "foreground_area_cv": round(float(values.std() / values.mean()), 4),
        "total_foreground_pixels": round(float(values.sum()), 2),
    }


def _grid_section(x, y, size, rows=3, cols=3):
    return min(rows - 1, int(y * rows / size)) * cols + min(cols - 1, int(x * cols / size)) + 1


def _controlled_specs(rng, n, target_count, kind, shapes):
    """Choose labels independently of the shuffled foreground area weights."""
    colors = [name for name, _ in scenes.BRIGHT_COLORS_NAMED]
    target_shape = rng.choice(scenes.SAFE_CONJUNCTION_SHAPES if kind == "conjunction" else shapes)
    target_color = rng.choice(colors)
    other_shapes = [shape for shape in shapes if shape != target_shape]
    other_colors = [color for color in colors if color != target_color]
    specs = []
    for slot in range(n):
        target = slot < target_count
        if kind == "shape":
            shape = target_shape if target else rng.choice(other_shapes)
            color = rng.choice(colors)
        elif kind == "color":
            shape = rng.choice(shapes)
            color = target_color if target else rng.choice(other_colors)
        else:
            # Include each difficult one-attribute decoy even in easy scenes.
            decoy = (slot - target_count) % 3
            shape = target_shape if target or decoy == 0 else rng.choice(other_shapes)
            color = target_color if target or decoy == 1 else rng.choice(other_colors)
        specs.append(dict(shape=shape, color=color, target=target))
    rng.shuffle(specs)
    return specs, {"shape": target_shape, "color": target_color}


def make_scene_problem(rng, size, spec):
    """Use the base scene renderers and predicates with six internal scene/target profiles."""
    difficulty, family = spec["difficulty"], spec["material"]
    kind, pool = spec.get("attribute_kind"), spec["pool"]
    if difficulty not in COUNT_RANGES or family not in ("geometry", "sticks"):
        raise ValueError("unsupported main material or difficulty")
    tier_index = tuple(COUNT_RANGES).index(difficulty)
    local = _private_scenes()
    # No separate diamond label: rotation turns the same regular square into it.
    local["GEOM_SHAPES"] = tuple(shape for shape in scenes.GEOM_SHAPES if shape != "diamond")
    local.update(
        COUNT_RANGES=dict(COUNT_RANGES),
        OVERLAP_CAPS=PAIR_CAP.copy(),
        SIZE_FRAC={tier: (.14, .20) for tier in COUNT_RANGES},
        CLUSTER_SPREAD_FRAC={tier: (None if tier == "easy" else .17) for tier in COUNT_RANGES},
        CAP_BIND_FLOOR={tier: 0.0 for tier in COUNT_RANGES},
        MIN_VISIBLE_FRACTION=VISIBLE_FLOOR[difficulty],
        SAME_ATTRIBUTE_OVERLAP_CAP=min(.18, PAIR_CAP[difficulty]),
    )
    # This independent RNG draws the budget before the count and target labels.
    size_rng = random.Random(rng.getrandbits(128))
    fraction = size_rng.uniform(*(GEOMETRY_BUDGET if family == "geometry" else LINE_BUDGET)[difficulty])
    requested_budget = fraction * size * size
    if difficulty == "ood":
        phase = rng.uniform(0, math.pi / 4)
        local["_cluster_centers"] = lambda _rng, canvas, _tier: [
            (canvas * (.5 + .31 * math.cos(phase + i * math.pi / 4)),
             canvas * (.5 + .31 * math.sin(phase + i * math.pi / 4))) for i in range(8)]
        local["CLUSTER_SPREAD_FRAC"][difficulty] = .085
    n = rng.randint(*COUNT_RANGES[difficulty])
    target_count = rng.randint(*TARGET_RANGES[difficulty]) if pool != "total" else n
    target_count = min(target_count, n - 3) if pool != "total" else n
    scene_params = (n, max(3.5, 8 - tier_index), min(10, 5 + tier_index), .45)
    weights = _weights(size_rng, n)
    audit = dict(policy=POLICY, budget_sampled_before_count_and_query=True,
                 requested_foreground_fraction=round(fraction, 6),
                 requested_foreground_pixels=round(requested_budget, 2),
                 weight_max_min_ratio=32.0, budget_depends_on_answer=False,
                 budget_is_tier_specific=False)
    target_labels = None
    fixed_specs = None
    if family == "geometry" and kind in ("shape", "color", "conjunction"):
        fixed_specs, target_labels = _controlled_specs(rng, n, target_count, kind, local["GEOM_SHAPES"])

    if family == "geometry":
        original_factory = local["_geometry_factory"]

        def factory(factory_rng, canvas, lo, hi, **kwargs):
            if fixed_specs is not None:
                kwargs["fixed_specs"] = fixed_specs
            make, entries, desc, schedule = original_factory(factory_rng, canvas, lo, hi, **kwargs)
            ss = kwargs.get("ss", 2)
            sprites = [make(factory_rng, slot) for slot in range(n)]
            initial_areas, lower, upper = [], [], []
            minimum_span, maximum_span = canvas / 16, canvas * .40
            for tile, _ in sprites:
                mask = np.asarray(tile)[..., 3] > 40
                ys, xs = np.where(mask)
                area = mask.sum() / (ss * ss)
                span = max(np.ptp(xs) + 1, np.ptp(ys) + 1) / ss
                initial_areas.append(area)
                lower.append(area * (minimum_span / span) ** 2)
                upper.append(area * (maximum_span / span) ** 2)
            areas, feasible_budget = _allocate(requested_budget, weights, lower, upper)
            adjusted, actual_areas, actual_spans = [], [], []
            for (tile, info), before, target in zip(sprites, initial_areas, areas):
                factor = math.sqrt(target / before)
                tile = tile.resize((max(4, round(tile.width * factor)), max(4, round(tile.height * factor))), Image.Resampling.LANCZOS)
                mask = np.asarray(tile)[..., 3] > 40
                ys, xs = np.where(mask)
                area = mask.sum() / (ss * ss)
                footprint = [(np.ptp(xs) + 1) / ss, (np.ptp(ys) + 1) / ss]
                span = max(footprint)
                info = {**info, "transform": {**info["transform"], "ink_budget_scale": round(factor, 6),
                        "foreground_area_px": round(float(area), 2), "foreground_span_px": round(float(span), 2)},
                        "foreground_area_px": round(float(area), 2),
                        "foreground_size_px": [round(float(value), 2) for value in footprint]}
                adjusted.append((tile, info))
                actual_areas.append(area)
                actual_spans.append(span)
            # Complete large objects are placed first; size was assigned before
            # every target label was matched and remains independent of targets.
            adjusted.sort(key=lambda item: -item[1]["foreground_area_px"])
            actual_areas = [item[1]["foreground_area_px"] for item in adjusted]
            actual_spans = [item[1]["transform"]["foreground_span_px"] for item in adjusted]
            audit.update(_area_stats(actual_areas))
            audit.update(feasible_foreground_pixels=round(feasible_budget, 2),
                         budget_clamped=abs(feasible_budget - requested_budget) > 1,
                         foreground_spans_px=[round(float(value), 2) for value in actual_spans],
                         minimum_span_px=round(float(min(actual_spans)), 2),
                         maximum_span_px=round(float(max(actual_spans)), 2))
            desc["size_policy"] = POLICY
            return lambda _rng, slot: adjusted[slot], entries, desc, schedule

        local["_geometry_factory"] = factory
        # The base total render path supplies all placements; the explicit
        # shape/color predicate below chooses exactly the planned labels.
        image, question, answer, cot, meta, elements = local["make_problem"](
            rng, size, difficulty=difficulty, family=family, scene_params=scene_params)
        if kind in ("shape", "color", "conjunction"):
            selected = [item for item in meta["placements"] if item["info"]["target"]]
            shape, color = target_labels["shape"], target_labels["color"]
            if kind == "shape":
                target, label = shape, scenes.SHAPE_NOUN[shape]
                question = f"How many {label} are there in the image?"
            elif kind == "color":
                target, label = color, f"{color} shapes"
                question = f"How many {label} are there in the image?"
            else:
                target, label = target_labels, f"{color} {scenes.SHAPE_NOUN[shape]}"
                question = f"How many {label} are in the image?"
            answer = str(len(selected))
            cot = f"Count only the {label}, ignoring the other objects. There are {answer}. The answer is {answer}."
            elements = [{"cx": item["centroid"][0], "cy": item["centroid"][1]} for item in selected]
            meta.update(subset_target=target, subset_count=len(selected), subset_kind="shape_color" if kind == "conjunction" else kind,
                        variant="shape_color_conjunction" if kind == "conjunction" else "subset")
        meta["shape_label_policy"] = "rotation_invariant_square_label_no_separate_diamond"
    else:
        image, question, answer, cot, meta, elements = _make_lines(
            rng, size_rng, size, difficulty, pool, kind, local, scene_params,
            n, target_count, requested_budget, weights, audit)

    audit["realized_foreground_fraction"] = round(audit["total_foreground_pixels"] / (size * size), 6)
    meta.update(size_audit=audit, size_policy=POLICY, difficulty=difficulty,
                difficulty_controls={"count_range": list(COUNT_RANGES[difficulty]),
                    "target_count_range": list(TARGET_RANGES[difficulty]) if pool != "total" else list(COUNT_RANGES[difficulty]),
                    "planned_count": n, "planned_target_count": target_count,
                    "counted_target_count": int(answer), "min_visible_fraction": VISIBLE_FLOOR[difficulty],
                    "pair_overlap_cap": PAIR_CAP[difficulty],
                    "region_union_size": len(meta.get("target_sections", [])),
                    "layout": "heldout_annular_clusters" if difficulty == "ood" and pool != "region" else "heldout_3x4_grid" if difficulty == "ood" else "main_scatter"},
                ood_axes=(["count_extrapolation", "3x4_region_grid"] if pool == "region" else ["count_extrapolation", "annular_layout"]) if difficulty == "ood" else [])
    if len(elements) != int(answer) or int(answer) != target_count:
        raise RuntimeError("main adapter did not realize the exact planned count predicate")
    return image, question, answer, cot, meta, elements


def _make_lines(rng, size_rng, size, difficulty, pool, kind, local, scene_params,
                n, target_count, requested_budget, weights, audit):
    """Base-scene strokes, with explicit targets and visible complete line objects."""
    rows, cols = (3, 4) if difficulty == "ood" else (3, 3)
    target_sections = sorted(rng.sample(range(1, rows * cols + 1), REGION_COUNT[difficulty])) if pool == "region" else []
    target_slots = set(rng.sample(range(n), target_count)) if pool != "total" else set(range(n))
    non_target_sections = [cell for cell in range(1, rows * cols + 1) if cell not in target_sections]
    draw_line = local["_draw_line_stroke"]
    captured = []

    def place_lines(scene_rng, canvas, count, min_gap, n_colors, striped_fraction):
        palette = scene_rng.sample(local["STICK_COLORS"], min(n_colors, len(local["STICK_COLORS"])))
        colors = [scene_rng.choice(palette) for _ in range(count)]
        color_counts = collections.Counter(colors)
        outlined = [color_counts[color] >= local["SAME_COLOR_OUTLINE_MIN_COUNT"] for color in colors]
        min_width, max_width = canvas * .0123, canvas * .08
        min_length, max_length = canvas * .070, canvas * .47
        outlines = [2 * local["LINE_OUTLINE_WIDTH_PX"] if value else 0 for value in outlined]
        lower = [min_length * (min_width + outline) + math.pi * (min_width + outline) ** 2 / 4 for outline in outlines]
        upper = [max_length * (max_width + outline) + math.pi * (max_width + outline) ** 2 / 4 for outline in outlines]
        areas, feasible_budget = _allocate(requested_budget, weights, lower, upper)
        dimensions = []
        for outline, target in zip(outlines, areas):
            def width_for(length):
                return 2 * (math.sqrt(length * length + math.pi * target) - length) / math.pi - outline
            low = max(min_width, width_for(max_length))
            high = max(low, min(max_width, width_for(min_length)))
            width = math.exp(size_rng.uniform(math.log(low), math.log(high)))
            effective = width + outline
            dimensions.append((target / effective - math.pi * effective / 4, width))
        # Long strokes are harder to place; shuffle equal-size ties independently
        # of every answer flag. Output is restored to canonical slot order.
        order = sorted(range(count), key=lambda slot: -areas[slot])
        curve_slots = set(scene_rng.sample(range(count), count // 2))
        lines, masks, remaining = [], [], []
        for slot in order:
            length, width = dimensions[slot]
            style = "candy" if (kind == "striped" and slot in target_slots) else scene_rng.choice(("plain", "tipped")) if kind == "striped" else scene_rng.choice(("plain", "plain", "tipped", "candy"))
            section_choices = target_sections if slot in target_slots else non_target_sections
            for _ in range(2500):
                if pool == "region":
                    cell = scene_rng.choice(section_choices) - 1
                    cx = canvas * ((cell % cols + scene_rng.uniform(.05, .95)) / cols)
                    cy = canvas * ((cell // cols + scene_rng.uniform(.05, .95)) / rows)
                else:
                    cx, cy = scene_rng.uniform(.06, .94) * canvas, scene_rng.uniform(.06, .94) * canvas
                    if difficulty == "ood" and not .19 * canvas <= math.hypot(cx - canvas / 2, cy - canvas / 2) <= .44 * canvas:
                        continue
                angle = scene_rng.uniform(0, math.pi)
                dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
                p0, p1 = (cx - dx, cy - dy), (cx + dx, cy + dy)
                control = None
                bend = 0.0
                points = [p0, p1]
                if slot in curve_slots:
                    bend = scene_rng.uniform(*local["CURVE_CONTROL_OFFSET_RANGE"])
                    offset = length * bend * scene_rng.choice((-1, 1))
                    control = (cx - math.sin(angle) * offset, cy + math.cos(angle) * offset)
                    points = local["_quadratic_curve_points"](p0, control, p1)
                    midpoint = points[len(points) // 2]
                    # The base scenes define a curve's center at its path midpoint.
                    shift = (cx - midpoint[0], cy - midpoint[1])
                    points = [(x + shift[0], y + shift[1]) for x, y in points]
                    p0, p1 = points[0], points[-1]
                    control = (control[0] + shift[0], control[1] + shift[1])
                padding = (width + outlines[slot]) / 2 + 2
                if any(not padding <= x <= canvas - padding or not padding <= y <= canvas - padding for x, y in points):
                    continue
                if any(math.hypot(cx - other["cx"], cy - other["cy"]) < 9 for other in lines):
                    continue
                stick = dict(slot=slot, p0=p0, p1=p1, ang=angle, length=length, width=width, color=colors[slot],
                             style=style, cx=cx, cy=cy, outlined=outlined[slot], area_weight=float(weights[slot]),
                             path_type="curve" if control is not None else "straight", control_point=control,
                             curve_control_offset_fraction=bend, path_points=points)
                tile = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
                draw_line(ImageDraw.Draw(tile), stick, ss=1)
                mask = np.asarray(tile)[..., 3] > 40
                area = int(mask.sum())
                accepted = True
                for old_mask, visible in zip(masks, remaining):
                    overlap = int(np.logical_and(mask, old_mask).sum())
                    if overlap / min(area, int(old_mask.sum())) > PAIR_CAP[difficulty] or (visible & ~mask).sum() < (VISIBLE_FLOOR[difficulty] + .015) * old_mask.sum():
                        accepted = False
                        break
                if not accepted:
                    continue
                for visible in remaining:
                    visible &= ~mask
                lines.append(stick)
                masks.append(mask)
                remaining.append(mask.copy())
                break
            else:
                # The base scenes retry the complete requested scene, never drop slots.
                return lines
        # Preserve the placement order: visibility checks assume this draw order.
        audit.update(feasible_foreground_pixels=round(feasible_budget, 2),
                     budget_clamped=abs(feasible_budget - requested_budget) > 1)
        return lines

    def collect_line(draw, stick, *, ss=2):
        captured.append(stick)
        draw_line(draw, stick, ss=ss)

    local["_draw_line_stroke"] = collect_line
    image, question, answer, cot, meta, elements = local["make_problem"](
        rng, size, difficulty=difficulty, family="sticks", scene_params=scene_params, place_lines=place_lines)
    # Render the chosen grid explicitly (the OOD columns differ from the base 3x3 grid).
    if pool == "region":
        image = Image.new("RGB", (size, size), rng.choice(local["BG_CHOICES"]))
        grid = ImageDraw.Draw(image)
        for col in range(1, cols):
            x = round(size * col / cols)
            grid.line((x, 0, x, size), fill=local["GRID_COLOR"], width=1)
        for row in range(1, rows):
            y = round(size * row / rows)
            grid.line((0, y, size, y), fill=local["GRID_COLOR"], width=1)
        # Visible labels make multi-region questions readable without mental indexing.
        for cell in range(1, rows * cols + 1):
            x, y = (cell - 1) % cols * size / cols + 4, (cell - 1) // cols * size / rows + 2
            grid.text((x, y), str(cell), font=load_bundled_font(18, "sans-bold"), fill=(115, 115, 130))
        layer = Image.new("RGBA", (size * 2, size * 2))
        for stick in captured:
            draw_line(ImageDraw.Draw(layer), stick, ss=2)
        layer = layer.resize((size, size), Image.Resampling.LANCZOS)
        image.paste(layer, (0, 0), layer)
    areas, masks = [], []
    for stick, item in zip(captured, meta["sticks"]):
        tile = Image.new("RGBA", (size * 2, size * 2))
        draw_line(ImageDraw.Draw(tile), stick, ss=2)
        mask = np.asarray(tile)[..., 3] > 40
        area = mask.sum() / 4
        ys, xs = np.where(mask)
        masks.append(mask)
        areas.append(area)
        section = _grid_section(stick["cx"], stick["cy"], size, rows, cols)
        stick["section"] = section
        item.update(section=section, centroid=[round(stick["cx"], 3), round(stick["cy"], 3)],
                    width_px=round(stick["width"], 4), length_px=round(stick["length"], 4),
                    foreground_area_px=round(float(area), 2),
                    foreground_bbox_px=[round(float(xs.min() / 2), 2), round(float(ys.min() / 2), 2),
                                        round(float((xs.max() + 1) / 2), 2), round(float((ys.max() + 1) / 2), 2)])
    union = np.zeros_like(masks[0])
    visible_fractions = []
    for mask in reversed(masks):
        visible_fractions.append(float((mask & ~union).sum() / mask.sum()))
        union |= mask
    visible_fractions.reverse()
    for item, visible in zip(meta["sticks"], visible_fractions):
        item["visible_fraction"] = round(visible, 6)
    if min(visible_fractions) < VISIBLE_FLOOR[difficulty]:
        raise RuntimeError("line antialiasing reduced visibility below the tier limit")
    audit.update(_area_stats(areas))
    audit.update(minimum_width_px=round(min(stick["width"] for stick in captured), 3),
                 maximum_width_px=round(max(stick["width"] for stick in captured), 3),
                 minimum_length_px=round(min(stick["length"] for stick in captured), 3),
                 maximum_length_px=round(max(stick["length"] for stick in captured), 3),
                 union_foreground_pixels=round(float(union.sum() / 4), 2))
    meta.update(minimum_visible_fraction=min(visible_fractions), minimum_allowed_visible_fraction=VISIBLE_FLOOR[difficulty])
    if pool == "region":
        elements = [stick for stick in captured if stick["section"] in target_sections]
        label = ", ".join(map(str, target_sections))
        noun = "section" if len(target_sections) == 1 else "sections"
        question = (f"The image has a {rows} by {cols} grid with sections numbered left to right, top to bottom. "
                    f"How many straight or curved lines have their center in {noun} {label}? "
                    "For a curved line, use its path midpoint. Count each line once.")
        answer = str(len(elements))
        cot = f"Count the lines whose path midpoint is in {noun} {label}. There are {answer}. The answer is {answer}."
        meta.update(variant="section", target_sections=target_sections, target_section=target_sections[0] if len(target_sections) == 1 else None,
                    grid_rows=rows, grid_cols=cols, section_counts=dict(collections.Counter(stick["section"] for stick in captured)),
                    region_rule="union_of_numbered_sections", region_center_definition="path_midpoint")
    elif kind == "striped":
        elements = [stick for stick in captured if stick["style"] == "candy"]
        answer = str(len(elements))
        question = "How many striped lines are shown? Ignore the plain and white-tipped lines."
        cot = f"Count only striped straight or curved lines. There are {answer}. The answer is {answer}."
        meta.update(variant="striped_subset", subset_count=len(elements))
    return image, question, answer, cot, meta, elements
