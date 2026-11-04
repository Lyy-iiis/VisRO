"""Base scene generator of count_clusters, in the style of BabyVision's Count Clusters task.

It draws geometry piles and straight or curved lines and asks for their total
count. Its design:

* geometry pair-overlap caps are 20% / 30% / 35%;
* geometry scenes enable synthetic outlines with 50% probability, fixed before
  placement retries; disabled scenes have no outlines, while enabled hard
  scenes outline every object and enabled easy/medium scenes outline only
  repeated-color groups;
* every geometry scene has guaranteed small/medium/large objects; overall
  size is sampled separately from rotation;
* hard geometry scenes use four or five broad groups, cap overlap degree
  at two neighbours, and keep at least 70% of every object visible;
* repeated-color lines are outlined, and a curved line's center is its path
  midpoint;
* unambiguous count-every-copy wording.

scene_ink_budget.py runs :func:`make_problem` in a private copy of this module
with its own tier tables, sizes and line placement, and adds the subset and
region questions; the release tiers (difficulty.py) set the object counts.
"""
from __future__ import annotations

import collections
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

COUNT_RANGES = {
    "easy": (1, 12),
    "medium": (13, 22),
    "hard": (23, 35),
}
OVERLAP_CAPS = {
    "easy": 0.20,
    "medium": 0.30,
    "hard": 0.35,
}
SIZE_SCALE_BANDS = {
    "small": (0.70, 0.82),
    "medium": (0.95, 1.05),
    "large": (1.18, 1.30),
}
# Keep object footprint stable across tiers so increasing N really increases
# density instead of being cancelled by progressively smaller sprites.
SIZE_FRAC = {difficulty: (0.14, 0.20) for difficulty in COUNT_RANGES}
CLUSTER_SPREAD_FRAC = {"easy": None, "medium": 0.13, "hard": 0.14}
CAP_BIND_FLOOR = {"easy": 0.0, "medium": 0.12, "hard": 0.0}
MAX_OVERLAP_NEIGHBORS = 2
MIN_VISIBLE_FRACTION = 0.70
SAME_ATTRIBUTE_OVERLAP_CAP = 0.18
OVERLAP_EDGE_THRESHOLD = 0.01
SAME_COLOR_OUTLINE_MIN_COUNT = 4
GEOMETRY_OUTLINE_WIDTH_PX = 2
GEOMETRY_OUTLINE_RGBA = (38, 42, 48, 255)
GEOMETRY_OUTLINE_SCENE_PROBABILITY = 0.50
CURVE_CONTROL_OFFSET_RANGE = (0.14, 0.26)
CURVE_SAMPLE_COUNT = 33
LINE_OUTLINE_WIDTH_PX = 2
LINE_OUTLINE_RGB = (38, 42, 48)
SAFE_CONJUNCTION_SHAPES = ("triangle", "pentagon", "star", "bar")
STICK_COLORS = (
    (220, 50, 47),
    (38, 139, 210),
    (133, 153, 0),
    (211, 130, 30),
    (108, 113, 196),
    (42, 161, 152),
    (211, 54, 130),
    (181, 137, 0),
    (60, 60, 60),
    (160, 80, 200),
)
BG_CHOICES = (
    (245, 245, 248),
    (250, 248, 240),
    (240, 247, 250),
    (248, 242, 245),
)
GRID_COLOR = (170, 170, 180)
BRIGHT_COLORS_NAMED = (
    ("red", (220, 45, 45)),
    ("orange", (240, 140, 30)),
    ("yellow", (235, 200, 40)),
    ("green", (50, 175, 75)),
    ("teal", (20, 160, 165)),
    ("blue", (40, 95, 205)),
    ("purple", (150, 70, 200)),
    ("pink", (235, 100, 165)),
    ("brown", (140, 90, 45)),
    ("gray", (110, 115, 125)),
)
GEOM_SHAPES = (
    "circle",
    "ellipse",
    "square",
    "diamond",
    "triangle",
    "pentagon",
    "star",
    "bar",
)
SHAPE_NOUN = {
    "circle": "circles",
    "ellipse": "ovals",
    "square": "squares",
    "diamond": "diamonds",
    "triangle": "triangles",
    "pentagon": "pentagons",
    "star": "stars",
    "bar": "rectangles",
}
GEOMETRY_TOTAL_QUESTIONS = (
    "How many separate shapes are shown altogether? Count every object.",
    "Count every separate geometric object in the image. How many are shown?",
)


def section_of(center_x: float, center_y: float, size: int) -> int:
    """Return the 1..9 row-major grid section containing a centroid."""
    column = min(2, int(center_x / (size / 3)))
    row = min(2, int(center_y / (size / 3)))
    return row * 3 + column + 1


def _poly(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    radius: float,
    sides: int,
    fill,
    *,
    rot: float = 0.0,
    inner: float | None = None,
) -> None:
    points = []
    for index in range(sides * (2 if inner else 1)):
        angle = rot + index * math.pi / (
            sides if inner else sides / 2
        )
        selected_radius = (
            radius
            if inner is None or index % 2 == 0
            else inner
        )
        points.append((
            center_x + selected_radius * math.cos(angle),
            center_y + selected_radius * math.sin(angle),
        ))
    draw.polygon(points, fill=fill)


def _size_tier_schedule(rng: random.Random, n: int) -> list[str]:
    """Return a shuffled, near-balanced small/medium/large assignment."""
    if n < 1:
        raise ValueError("size schedule requires at least one object")
    tiers = tuple(SIZE_SCALE_BANDS)
    offset = rng.randrange(len(tiers))
    schedule = [tiers[(offset + index) % len(tiers)] for index in range(n)]
    rng.shuffle(schedule)
    return schedule


def _uniform_transform_fit(
    sprite: Image.Image,
    rng: random.Random,
    *,
    size_tier: str,
):
    """Shape-label-safe uniform scaling followed by arbitrary rotation."""
    if size_tier not in SIZE_SCALE_BANDS:
        raise ValueError(f"unknown size tier: {size_tier!r}")
    scale = rng.uniform(*SIZE_SCALE_BANDS[size_tier])
    angle = rng.uniform(0.0, 360.0)
    width = max(1, int(round(sprite.width * scale)))
    height = max(1, int(round(sprite.height * scale)))
    sprite = sprite.resize((width, height), Image.Resampling.LANCZOS)
    sprite = sprite.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    transform = {
        "size_tier": size_tier,
        "size_scale": round(scale, 6),
        "aspect_ratio": 1.0,
        "scale_x": round(scale, 6),
        "scale_y": round(scale, 6),
        "angle_degrees": round(angle, 6),
    }
    return sprite, transform


def _apply_outline(
    sprite: Image.Image,
    *,
    width_px: int,
    color: tuple[int, int, int, int],
    ss: int,
) -> Image.Image:
    """Add a solid external outline around an RGBA sprite."""
    padding = width_px * ss
    padded = Image.new(
        "RGBA",
        (sprite.width + 2 * padding, sprite.height + 2 * padding),
        (0, 0, 0, 0),
    )
    padded.alpha_composite(sprite, (padding, padding))
    sprite = padded
    alpha = sprite.getchannel("A")
    dilation_size = 2 * width_px * ss + 1
    outline_alpha = alpha.filter(ImageFilter.MaxFilter(dilation_size))
    result = Image.new("RGBA", sprite.size, color)
    result.putalpha(outline_alpha)
    result.alpha_composite(sprite)
    return result


def _geometry_outline_scene_enabled(rng: random.Random) -> bool:
    """Enable all synthetic geometry outlines for half of scenes."""
    return rng.random() < GEOMETRY_OUTLINE_SCENE_PROBABILITY


def _geometry_factory(
    rng: random.Random,
    size: int,
    lo: float,
    hi: float,
    *,
    n: int | None = None,
    color_count: int | None = None,
    fixed_specs: list[dict] | None = None,
    outlined: bool = False,
    outline_scene_enabled: bool = True,
    size_tiers: list[str] | None = None,
    ss: int = 2,
):
    if n is None:
        n = len(fixed_specs) if fixed_specs is not None else 1
    if n < 1:
        raise ValueError("geometry factory requires at least one object")
    if fixed_specs is not None and len(fixed_specs) != n:
        raise ValueError("fixed geometry specifications must match n")
    if size_tiers is None:
        size_tiers = _size_tier_schedule(rng, n)
    if len(size_tiers) != n:
        raise ValueError("geometry size-tier schedule must match n")
    if fixed_specs is None:
        palette_size = min(
            color_count or rng.randint(3, 5), len(BRIGHT_COLORS_NAMED)
        )
        palette = rng.sample(BRIGHT_COLORS_NAMED, palette_size)
        shapes = rng.sample(GEOM_SHAPES, rng.randint(3, 5))
        render_specs = [
            {
                "shape": rng.choice(shapes),
                "color": rng.choice(palette)[0],
                "target": False,
            }
            for _ in range(n)
        ]
    else:
        color_names = sorted({spec["color"] for spec in fixed_specs})
        color_lookup = dict(BRIGHT_COLORS_NAMED)
        palette = [(name, color_lookup[name]) for name in color_names]
        shapes = sorted({spec["shape"] for spec in fixed_specs})
        render_specs = fixed_specs
    color_counts = collections.Counter(spec["color"] for spec in render_specs)
    if outlined and not outline_scene_enabled:
        raise ValueError("global geometry outlines require an enabled scene gate")
    repeated_outline_colors = (
        {
            color for color, count in color_counts.items()
            if count >= SAME_COLOR_OUTLINE_MIN_COUNT
        }
        if outline_scene_enabled else set()
    )
    any_outline = outlined or bool(repeated_outline_colors)
    desc = {
        "submode": "geometry",
        "palette": [color[0] for color in palette],
        "shape_set": list(shapes),
        "size_tier_counts": dict(collections.Counter(size_tiers)),
        "outline_scene_enabled": outline_scene_enabled,
        "outline_scene_probability": GEOMETRY_OUTLINE_SCENE_PROBABILITY,
        "global_outline_enabled": outlined,
        "repeated_color_outline_enabled": bool(repeated_outline_colors),
        "outline_enabled": any_outline,
        "outline_width_px": GEOMETRY_OUTLINE_WIDTH_PX if any_outline else 0,
        "outlined_colors": sorted(
            set(color_counts) if outlined else repeated_outline_colors
        ),
        "color_counts": dict(sorted(color_counts.items())),
    }
    base_side = max(8, int(size * rng.uniform(lo, hi)) * ss)
    cache = {}

    def make(copy_rng, slot):
        if slot in cache:
            return cache[slot]
        spec = render_specs[slot]
        shape = spec["shape"]
        color_name = spec["color"]
        color_rgb = dict(palette)[color_name]
        is_target = bool(spec.get("target", False))
        color = color_rgb + (255,)
        side = base_side
        tile = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        center, radius = side / 2, side * 0.46
        if shape == "circle":
            draw.ellipse(
                [center - radius, center - radius, center + radius, center + radius],
                fill=color,
            )
        elif shape == "ellipse":
            draw.ellipse(
                [
                    center - radius,
                    center - radius * 0.6,
                    center + radius,
                    center + radius * 0.6,
                ],
                fill=color,
            )
        elif shape == "square":
            draw.rectangle(
                [
                    center - radius * 0.8,
                    center - radius * 0.8,
                    center + radius * 0.8,
                    center + radius * 0.8,
                ],
                fill=color,
            )
        elif shape == "bar":
            draw.rounded_rectangle(
                [
                    center - radius,
                    center - radius * 0.33,
                    center + radius,
                    center + radius * 0.33,
                ],
                radius=radius * 0.3,
                fill=color,
            )
        elif shape == "diamond":
            _poly(draw, center, center, radius, 4, color, rot=0.0)
        elif shape == "triangle":
            _poly(
                draw, center, center, radius, 3, color, rot=-math.pi / 2
            )
        elif shape == "pentagon":
            _poly(
                draw, center, center, radius, 5, color, rot=-math.pi / 2
            )
        else:
            _poly(
                draw,
                center,
                center,
                radius,
                5,
                color,
                rot=-math.pi / 2,
                inner=radius * 0.45,
            )
        object_outlined = outlined or color_name in repeated_outline_colors
        sprite, transform = _uniform_transform_fit(
            tile, copy_rng, size_tier=size_tiers[slot]
        )
        if object_outlined:
            sprite = _apply_outline(
                sprite,
                width_px=GEOMETRY_OUTLINE_WIDTH_PX,
                color=GEOMETRY_OUTLINE_RGBA,
                ss=ss,
            )
        result = sprite, {
            "shape": shape,
            "color": color_name,
            "target": is_target,
            "outlined": object_outlined,
            "transform": transform,
        }
        cache[slot] = result
        return result

    return make, None, desc, None


def _scatter(
    rng: random.Random,
    size: int,
    n: int,
    make_sprite,
    *,
    max_overlap: float,
    ss: int = 2,
    min_cdist_frac: float = 0.05,
    cluster_centers: list[tuple[float, float]] | None = None,
    cluster_schedule: list[int] | None = None,
    cluster_spread_frac: float = 0.12,
    same_attribute_overlap_cap: float | None = None,
    max_overlap_neighbors: int | None = None,
    min_visible_fraction: float = 0.0,
):
    if not 0.0 <= min_visible_fraction <= 1.0:
        raise ValueError("min_visible_fraction must be in [0, 1]")
    if max_overlap_neighbors is not None and max_overlap_neighbors < 0:
        raise ValueError("max_overlap_neighbors must be nonnegative")
    if cluster_schedule is not None:
        if cluster_centers is None or len(cluster_schedule) != n:
            raise ValueError("cluster schedule requires matching cluster centers")
        if any(not 0 <= index < len(cluster_centers) for index in cluster_schedule):
            raise ValueError("cluster schedule index is out of range")
    width = height = size * ss
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    placed = []
    attempts = 0
    max_attempts = n * 1000
    minimum_centroid_distance = size * min_cdist_frac
    while len(placed) < n and attempts < max_attempts:
        attempts += 1
        slot = len(placed)
        sprite, info = make_sprite(rng, slot)
        sprite_width, sprite_height = sprite.size
        if (
            sprite_width < 4
            or sprite_height < 4
            or sprite_width >= width
            or sprite_height >= height
        ):
            continue
        mask = np.asarray(sprite)[..., 3] > 40
        area = int(mask.sum())
        if area < 50:
            continue
        ys, xs = np.where(mask)
        cluster_index = None
        if cluster_centers:
            cluster_index = (
                cluster_schedule[slot]
                if cluster_schedule is not None
                else rng.randrange(len(cluster_centers))
            )
            anchor_x, anchor_y = cluster_centers[cluster_index]
            spread = size * cluster_spread_frac
            target_cx = rng.gauss(anchor_x, spread)
            target_cy = rng.gauss(anchor_y, spread)
            x = int(round(target_cx * ss - xs.mean()))
            y = int(round(target_cy * ss - ys.mean()))
            if not (0 <= x <= width - sprite_width and 0 <= y <= height - sprite_height):
                continue
        else:
            x = rng.randint(0, width - sprite_width)
            y = rng.randint(0, height - sprite_height)
        cx, cy = (x + xs.mean()) / ss, (y + ys.mean()) / ss
        if any(
            (cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2
            < minimum_centroid_distance ** 2
            for item in placed
        ):
            continue
        accepted = True
        current_neighbor_count = 0
        overlap_updates = []
        for item in placed:
            px, py, previous_mask = item["x"], item["y"], item["mask"]
            x0, y0 = max(x, px), max(y, py)
            x1 = min(x + sprite_width, px + previous_mask.shape[1])
            y1 = min(y + sprite_height, py + previous_mask.shape[0])
            if x0 >= x1 or y0 >= y1:
                continue
            current = mask[y0 - y:y1 - y, x0 - x:x1 - x]
            previous = previous_mask[y0 - py:y1 - py, x0 - px:x1 - px]
            overlap = int(np.logical_and(current, previous).sum())
            fraction = overlap / min(area, item["area"])
            pair_cap = max_overlap
            if (
                same_attribute_overlap_cap is not None
                and _shares_identity_or_color(info, item["info"])
            ):
                pair_cap = min(pair_cap, same_attribute_overlap_cap)
            if fraction > pair_cap:
                accepted = False
                break
            is_neighbor = fraction >= OVERLAP_EDGE_THRESHOLD
            if is_neighbor:
                current_neighbor_count += 1
                if (
                    max_overlap_neighbors is not None
                    and (
                        current_neighbor_count > max_overlap_neighbors
                        or item["overlap_neighbors"] >= max_overlap_neighbors
                    )
                ):
                    accepted = False
                    break
            previous_visible = item["visible_mask"][
                y0 - py:y1 - py, x0 - px:x1 - px
            ]
            covered_visible = np.logical_and(previous_visible, current)
            covered_count = int(covered_visible.sum())
            if (
                item["visible_area"] - covered_count
                < min_visible_fraction * item["area"]
            ):
                accepted = False
                break
            overlap_updates.append(
                {
                    "item": item,
                    "visible_slice": (
                        slice(y0 - py, y1 - py),
                        slice(x0 - px, x1 - px),
                    ),
                    "covered_visible": covered_visible,
                    "covered_count": covered_count,
                    "is_neighbor": is_neighbor,
                }
            )
        if not accepted:
            continue
        for update in overlap_updates:
            item = update["item"]
            if update["is_neighbor"]:
                item["overlap_neighbors"] += 1
            if update["covered_count"]:
                visible_view = item["visible_mask"][update["visible_slice"]]
                visible_view[update["covered_visible"]] = False
                item["visible_area"] -= update["covered_count"]
        layer.alpha_composite(sprite, (x, y))
        placed.append(
            {
                "x": x,
                "y": y,
                "mask": mask,
                "area": area,
                "visible_mask": mask.copy(),
                "visible_area": area,
                "overlap_neighbors": current_neighbor_count,
                "cluster_index": cluster_index,
                "cx": cx,
                "cy": cy,
                "info": info,
            }
        )
    return placed, layer.resize((size, size), Image.Resampling.LANCZOS)


def _shares_identity_or_color(first_info: dict, second_info: dict) -> bool:
    """Return whether a pair repeats the same icon identity or flat color."""
    for key in ("icon", "color"):
        first = first_info.get(key)
        second = second_info.get(key)
        if first is not None and second is not None and first == second:
            return True
    return False


def _pair_overlap_fraction(first: dict, second: dict) -> float:
    x0, y0 = max(first["x"], second["x"]), max(first["y"], second["y"])
    x1 = min(
        first["x"] + first["mask"].shape[1],
        second["x"] + second["mask"].shape[1],
    )
    y1 = min(
        first["y"] + first["mask"].shape[0],
        second["y"] + second["mask"].shape[0],
    )
    if x0 >= x1 or y0 >= y1:
        return 0.0
    first_view = first["mask"][
        y0 - first["y"]:y1 - first["y"],
        x0 - first["x"]:x1 - first["x"],
    ]
    second_view = second["mask"][
        y0 - second["y"]:y1 - second["y"],
        x0 - second["x"]:x1 - second["x"],
    ]
    overlap = int(np.logical_and(first_view, second_view).sum())
    return overlap / min(first["area"], second["area"])


def overlap_metrics(placed: list[dict]) -> dict:
    participants = set()
    edges = 0
    maximum = 0.0
    same_attribute_maximum = 0.0
    for index, first in enumerate(placed):
        for second_index, second in enumerate(placed[index + 1:], index + 1):
            fraction = _pair_overlap_fraction(first, second)
            maximum = max(maximum, fraction)
            if _shares_identity_or_color(first["info"], second["info"]):
                same_attribute_maximum = max(same_attribute_maximum, fraction)
            if fraction >= OVERLAP_EDGE_THRESHOLD:
                edges += 1
                participants.update((index, second_index))
    n = len(placed)
    visible_fractions = [
        item.get("visible_area", item["area"]) / item["area"]
        for item in placed
    ]
    return {
        "max_pair_overlap": round(maximum, 6),
        "max_same_attribute_pair_overlap": round(same_attribute_maximum, 6),
        "overlap_participation": round(len(participants) / n, 6) if n else 0.0,
        "overlap_edges_per_object": round(2 * edges / n, 6) if n else 0.0,
        "max_overlap_neighbors": max(
            (item.get("overlap_neighbors", 0) for item in placed), default=0
        ),
        "minimum_visible_fraction": round(min(visible_fractions), 6)
        if visible_fractions else 0.0,
    }


def _cluster_centers(rng: random.Random, size: int, difficulty: str):
    spread = CLUSTER_SPREAD_FRAC[difficulty]
    if spread is None:
        return None
    corners = [(0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80)]
    if difficulty == "medium":
        selected = corners
    else:
        anchors = corners + [(0.50, 0.50)]
        selected = rng.sample(anchors, rng.randint(4, 5))
    return [
        (
            size * min(0.84, max(0.16, x + rng.uniform(-0.035, 0.035))),
            size * min(0.84, max(0.16, y + rng.uniform(-0.035, 0.035))),
        )
        for x, y in selected
    ]


def _balanced_cluster_schedule(
    rng: random.Random, n: int, cluster_count: int
) -> list[int]:
    if n < cluster_count or cluster_count < 1:
        raise ValueError("each cluster requires at least one object")
    schedule = [index % cluster_count for index in range(n)]
    rng.shuffle(schedule)
    return schedule


def _make_pile_problem(
    rng: random.Random,
    size: int,
    difficulty: str,
    scene_params: tuple,
):
    n, min_gap, n_colors, _ = scene_params
    geometry_outline_scene_enabled = _geometry_outline_scene_enabled(rng)

    overlap_cap = OVERLAP_CAPS[difficulty]
    lo, hi = SIZE_FRAC[difficulty]
    min_cdist_frac = min_gap / 448.0
    required_overlap = CAP_BIND_FLOOR[difficulty]
    last_count = 0
    last_metrics = {"max_pair_overlap": 0.0}
    successful_centers = None
    successful_cluster_schedule = None
    for _scene_attempt in range(48):
        centers = _cluster_centers(rng, size, difficulty)
        cluster_schedule = (
            _balanced_cluster_schedule(rng, n, len(centers)) if centers else None
        )
        make, _, desc, _ = _geometry_factory(
            rng,
            size,
            lo,
            hi,
            n=n,
            color_count=n_colors,
            outlined=(
                difficulty == "hard" and geometry_outline_scene_enabled
            ),
            outline_scene_enabled=geometry_outline_scene_enabled,
        )
        placed, layer = _scatter(
            rng,
            size,
            n,
            make,
            max_overlap=overlap_cap,
            min_cdist_frac=min_cdist_frac,
            cluster_centers=centers,
            cluster_schedule=cluster_schedule,
            cluster_spread_frac=CLUSTER_SPREAD_FRAC[difficulty] or 0.12,
            same_attribute_overlap_cap=SAME_ATTRIBUTE_OVERLAP_CAP,
            max_overlap_neighbors=MAX_OVERLAP_NEIGHBORS,
            min_visible_fraction=MIN_VISIBLE_FRACTION,
        )
        last_count = len(placed)
        last_metrics = overlap_metrics(placed)
        if len(placed) == n and last_metrics["max_pair_overlap"] >= required_overlap:
            successful_centers = centers
            successful_cluster_schedule = cluster_schedule
            break
    else:
        raise RuntimeError(
            f"could place only {last_count}/{n} objects after scene retries: "
            f"difficulty={difficulty} family=geometry metrics={last_metrics}"
        )

    image = Image.new("RGB", (size, size), (255, 255, 255))
    image.paste(layer, (0, 0), layer)
    meta = {
        "difficulty": difficulty,
        "family": "geometry",
        "n_requested": n,
        "n_total": len(placed),
        "overlap_cap": overlap_cap,
        "max_realized_pair_overlap": last_metrics["max_pair_overlap"],
        "overlap_participation": last_metrics["overlap_participation"],
        "overlap_edges_per_object": last_metrics["overlap_edges_per_object"],
        "max_overlap_neighbors": last_metrics["max_overlap_neighbors"],
        "maximum_allowed_overlap_neighbors": MAX_OVERLAP_NEIGHBORS,
        "minimum_visible_fraction": last_metrics["minimum_visible_fraction"],
        "minimum_allowed_visible_fraction": MIN_VISIBLE_FRACTION,
        "same_attribute_overlap_cap": SAME_ATTRIBUTE_OVERLAP_CAP,
        "max_same_attribute_pair_overlap": last_metrics[
            "max_same_attribute_pair_overlap"
        ],
        "required_overlap_floor": required_overlap,
        "min_centroid_distance_fraction": round(min_cdist_frac, 6),
        "cluster_centers": (
            [[round(x, 1), round(y, 1)] for x, y in successful_centers]
            if successful_centers else []
        ),
        "cluster_occupancy": (
            dict(sorted(collections.Counter(successful_cluster_schedule).items()))
            if successful_cluster_schedule else {}
        ),
        "size_fraction_range": [lo, hi],
        "n_colors": n_colors,
        **desc,
    }
    meta["placements"] = [
        {
            "centroid": [round(item["cx"], 1), round(item["cy"], 1)],
            "cluster_index": item["cluster_index"],
            "visible_fraction": round(item["visible_area"] / item["area"], 6),
            "overlap_neighbors": item["overlap_neighbors"],
            "info": item["info"],
        }
        for item in placed
    ]

    question = rng.choice(GEOMETRY_TOTAL_QUESTIONS)
    answer = str(len(placed))
    cot = (
        f"Scan the pile and segment each shape from its neighbours, "
        f"counting each exactly once. There are {len(placed)} in total. "
        f"The answer is {len(placed)}."
    )
    elements = [{"cx": item["cx"], "cy": item["cy"]} for item in placed]
    meta["variant"] = "total"
    return image, question, answer, cot, meta, elements


def _quadratic_curve_points(
    p0: tuple[float, float],
    control: tuple[float, float],
    p1: tuple[float, float],
    *,
    samples: int = CURVE_SAMPLE_COUNT,
) -> list[tuple[float, float]]:
    if samples < 2:
        raise ValueError("a curve requires at least two samples")
    points = []
    for index in range(samples):
        t = index / (samples - 1)
        one_minus_t = 1.0 - t
        points.append(
            (
                one_minus_t * one_minus_t * p0[0]
                + 2.0 * one_minus_t * t * control[0]
                + t * t * p1[0],
                one_minus_t * one_minus_t * p0[1]
                + 2.0 * one_minus_t * t * control[1]
                + t * t * p1[1],
            )
        )
    return points


def _polyline_sample(
    points: list[tuple[float, float]], distance: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return point and unit tangent at an arc-length position."""
    remaining = distance
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-9:
            continue
        if remaining <= segment_length:
            fraction = remaining / segment_length
            point = (
                start[0] + dx * fraction,
                start[1] + dy * fraction,
            )
            return point, (dx / segment_length, dy / segment_length)
        remaining -= segment_length
    start, end = points[-2], points[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    segment_length = max(1e-9, math.hypot(dx, dy))
    return end, (dx / segment_length, dy / segment_length)


def _draw_line_stroke(
    draw: ImageDraw.ImageDraw,
    stick: dict,
    *,
    ss: int,
) -> None:
    """Render a straight or gently curved rounded line in one of the stick styles."""
    points = [(x * ss, y * ss) for x, y in stick["path_points"]]
    width = max(2, int(round(stick["width"] * ss)))
    radius = width / 2.0
    color = tuple(stick["color"])

    if stick.get("outlined", False):
        outline_width = width + 2 * LINE_OUTLINE_WIDTH_PX * ss
        outline_radius = outline_width / 2.0
        draw.line(
            points,
            fill=LINE_OUTLINE_RGB + (255,),
            width=outline_width,
            joint="curve",
        )
        for cx, cy in (points[0], points[-1]):
            draw.ellipse(
                [
                    cx - outline_radius,
                    cy - outline_radius,
                    cx + outline_radius,
                    cy + outline_radius,
                ],
                fill=LINE_OUTLINE_RGB + (255,),
            )

    draw.line(points, fill=color + (255,), width=width, joint="curve")
    for cx, cy in (points[0], points[-1]):
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color + (255,),
        )

    if stick["style"] == "candy":
        segment_lengths = [
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(points, points[1:])
        ]
        total_length = sum(segment_lengths)
        stripe_step = max(6.0 * ss, width * 0.9)
        stripe_distance = stripe_step * 0.5
        stripe_width = max(2, int(width * 0.28))
        while stripe_distance < total_length:
            (mx, my), (tx, ty) = _polyline_sample(points, stripe_distance)
            nx, ny = -ty, tx
            draw.line(
                [
                    mx - nx * radius,
                    my - ny * radius,
                    mx + nx * radius,
                    my + ny * radius,
                ],
                fill=(255, 255, 255, 235),
                width=stripe_width,
            )
            stripe_distance += stripe_step
    elif stick["style"] == "tipped":
        cx, cy = points[0]
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, 255, 255, 255),
        )


LINE_TOTAL_QUESTIONS = (
    "How many separate lines are shown altogether? Count both straight and curved lines.",
    "Count every separate straight or curved line in the image. How many are shown?",
)


def _make_stick_problem(
    rng: random.Random,
    size: int,
    difficulty: str,
    scene_params: tuple,
    place_lines,
):
    n_requested, min_gap, n_colors, candy_frac = scene_params
    last_count = 0
    for _scene_attempt in range(24):
        sticks = place_lines(
            rng, size, n_requested, min_gap, n_colors, candy_frac
        )
        last_count = len(sticks)
        if last_count == n_requested:
            break
    else:
        raise RuntimeError(
            f"could place only {last_count}/{n_requested} sticks after scene "
            f"retries: difficulty={difficulty}"
        )

    line_color_counts = collections.Counter(stick["color"] for stick in sticks)
    outlined_line_colors = {
        color for color, count in line_color_counts.items()
        if count >= SAME_COLOR_OUTLINE_MIN_COUNT
    }
    for stick in sticks:
        stick["outlined"] = stick["color"] in outlined_line_colors

    background = rng.choice(BG_CHOICES)
    image = Image.new("RGB", (size, size), background)
    grid = ImageDraw.Draw(image)
    for k in (1, 2):
        position = int(size * k / 3)
        grid.line([position, 0, position, size], fill=GRID_COLOR, width=1)
        grid.line([0, position, size, position], fill=GRID_COLOR, width=1)

    ss = 2
    layer = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for stick in sticks:
        _draw_line_stroke(draw, stick, ss=ss)
    layer = layer.resize((size, size), Image.Resampling.LANCZOS)
    image.paste(layer, (0, 0), layer)

    section_counts = {index: 0 for index in range(1, 10)}
    for stick in sticks:
        section = section_of(stick["cx"], stick["cy"], size)
        stick["section"] = section
        section_counts[section] += 1
    question = rng.choice(LINE_TOTAL_QUESTIONS)
    answer = str(len(sticks))
    cot = (
        f"Scan the image and count each separate straight or curved line "
        f"exactly once. There are {len(sticks)} lines in total. The answer is "
        f"{len(sticks)}."
    )
    elements = sticks
    meta = {
        "difficulty": difficulty,
        "family": "sticks",
        "render_family": "mixed_lines",
        "variant": "total",
        "n_requested": n_requested,
        "n_total": len(sticks),
        "section_counts": section_counts,
        "min_gap": round(min_gap, 2),
        "candy_frac": round(candy_frac, 2),
        "n_colors": n_colors,
        "style_counts": dict(collections.Counter(stick["style"] for stick in sticks)),
        "path_type_counts": dict(collections.Counter(
            stick["path_type"] for stick in sticks
        )),
        "curved_line_fraction": round(
            sum(stick["path_type"] == "curve" for stick in sticks) / len(sticks),
            6,
        ),
        "outlined_colors": [list(color) for color in sorted(outlined_line_colors)],
        "subset_count": None,
        "sticks": [
            {
                "p0": [round(stick["p0"][0], 1), round(stick["p0"][1], 1)],
                "p1": [round(stick["p1"][0], 1), round(stick["p1"][1], 1)],
                "centroid": [round(stick["cx"], 1), round(stick["cy"], 1)],
                "section": stick["section"],
                "style": stick["style"],
                "color": list(stick["color"]),
                "path_type": stick["path_type"],
                "control_point": (
                    [round(value, 1) for value in stick["control_point"]]
                    if stick["control_point"] is not None else None
                ),
                "curve_control_offset_fraction": stick[
                    "curve_control_offset_fraction"
                ],
                "outlined": stick["outlined"],
            }
            for stick in sticks
        ],
    }
    return image, question, answer, cot, meta, elements


def make_problem(
    rng: random.Random,
    size: int,
    *,
    difficulty: str,
    family: str,
    scene_params: tuple,
    place_lines=None,
):
    """Draw a geometry or line scene and ask for its total count.

    ``scene_params`` is (object count, minimum gap, color count, striped
    fraction); ``place_lines`` places the line skeletons of a sticks scene.
    """
    if difficulty not in COUNT_RANGES:
        raise ValueError(f"unknown difficulty: {difficulty!r}")
    if family == "sticks":
        return _make_stick_problem(rng, size, difficulty, scene_params, place_lines)
    if family != "geometry":
        raise ValueError(f"unknown family: {family!r}")
    return _make_pile_problem(rng, size, difficulty, scene_params)
