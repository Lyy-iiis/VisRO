"""Six-profile mask-overlap adapter for the five-pool count-clusters mixture.

The pair predicate, groups-first placement and measured occlusion constraint
come from the overlap tier generator. Each tier specifies increasing target counts, longer
connected components, and a required measured maximum-occlusion band. The
``ood`` profile adds a degree-three branching component; it is the extended
half of xxhard in either asset split. Absolute
contact floors and clearances are recorded per tier, alongside the common
15% relative threshold. Object sizes are assigned before overlap labels from
a shared broad population with an answer-independent scene area budget.
Masks, rather than palette colors, define the answer. Fluent is sampled only
from the requested split, preserves its own shading, and uses a conservative
solid-object allowlist. Enclosed alpha holes get an opaque white backing: this
makes High Contrast outlines represent physical objects instead of transparent
wire frames. Actual openings connected to the exterior remain open.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .difficulty import COUNT_RANGES, OVERLAP_TARGET_RANGES

# The scene maximum occlusion must actually fall in its specified band.
# This is an achieved difficulty control, not just a progressively looser cap.
TIERS = {
    "easy": {"n": COUNT_RANGES["easy"], "gamma_band": (.23, .29), "clearance": 16., "intersection": 160., "min_area": 720., "component": (2, 3)},
    "medium": {"n": COUNT_RANGES["medium"], "gamma_band": (.31, .37), "clearance": 12., "intersection": 100., "min_area": 520., "component": (2, 3)},
    "hard": {"n": COUNT_RANGES["hard"], "gamma_band": (.40, .46), "clearance": 9., "intersection": 64., "min_area": 400., "component": (3, 4)},
    "xhard": {"n": COUNT_RANGES["xhard"], "gamma_band": (.48, .54), "clearance": 5., "intersection": 42., "min_area": 400., "component": (4, 5)},
    "xxhard": {"n": COUNT_RANGES["xxhard"], "gamma_band": (.57, .63), "clearance": 3., "intersection": 28., "min_area": 400., "component": (5, 7)},
    "ood": {"n": COUNT_RANGES["ood"], "gamma_band": (.64, .70), "clearance": 1.5, "intersection": 20., "min_area": 400., "component": (7, 9)},
}
for _name, _cfg in TIERS.items():
    _cfg["gamma"] = _cfg["gamma_band"][1]
    _cfg["edge_band"] = {"easy": (.23, .29), "medium": (.23, .34), "hard": (.24, .31), "xhard": (.27, .34), "xxhard": (.31, .37), "ood": (.35, .40)}[_name]
    _cfg["source_tier"] = _name
    _cfg["max_neighbors"] = 3 if _name == "ood" else 2
CLEAR_PX = 21.0  # defaults of the public mask helpers
OVERLAP_MIN = .15
OVERLAP_MIN_PX = 196
QUESTION = ("How many of the objects overlap at least one other object? "
            "(Answer with a single number.)")
PALETTE = ((204, 49, 69), (25, 112, 172), (32, 150, 98), (223, 146, 26),
           (131, 74, 180), (178, 77, 27), (18, 158, 164), (218, 87, 159))
# Solid items only: no water panels, light beams, paired accessories, literal
# holes (doughnuts/rings), steam, or objects whose noun needs counting_notes.
SOLID_FLUENT_CLASSES = frozenset("""
american_football amphora baby_bottle backpack baguette_bread banana baseball
basketball bell bell_pepper beverage_box briefcase broccoli bucket burrito
cactus canned_food candy carrot cheese_wedge cherry_blossom chess_pawn chestnut
chocolate_bar clapper_board clipboard clutch_bag coat coffin coin compass
computer_disk computer_mouse cookie croissant crown crystal_ball cucumber
cupcake curling_stone cut_of_meat dress dumpling ear_of_corn egg eggplant
fire_extinguisher fish_cake_with_swirl flat_shoe flatbread floppy_disk
fortune_cookie framed_picture fried_shrimp funeral_urn game_controller game_die
garlic gem_stone ginger_root glass_of_milk hamburger handbag headstone
hedgehog hiking_boot honey_pot hot_dog hot_pepper ice_cream ice_skate
ice_cream jack-o-lantern key kiwi_fruit label lemon light_bulb lipstick lobster
lollipop luggage mango melon microphone mobile_phone mushroom oil_drum onion
package pager peach pear penguin pie pill pineapple pool_8_ball postbox potato
poultry_leg printer purse puzzle_piece radio rice_ball rice_cracker rock
rugby_football running_shoe salt sandwich shield shorts soap soccer_ball
softball spiral_notepad sponge strawberry stuffed_flatbread t-shirt taco
takeout_box tangerine teddy_bear telephone telephone_receiver television
test_tube timer_clock tomato toolbox top_hat trackball trophy volleyball
waffle watch womans_boot wrapped_gift
""".split())


def _intersection(a_mask, ax, ay, b_mask, bx, by):
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + a_mask.shape[1], bx + b_mask.shape[1])
    y1 = min(ay + a_mask.shape[0], by + b_mask.shape[0])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.count_nonzero(
        a_mask[y0-ay:y1-ay, x0-ax:x1-ax]
        & b_mask[y0-by:y1-by, x0-bx:x1-bx]))


@dataclass
class Obj:
    tile: Image.Image
    x: int
    y: int
    desc: dict
    size: int = 448
    mask: np.ndarray | None = None
    clearance: float = CLEAR_PX
    intersection_floor: float = OVERLAP_MIN_PX
    area: int = field(init=False)

    def __post_init__(self):
        if self.mask is None:
            self.mask = ndimage.binary_fill_holes(np.asarray(self.tile.getchannel("A")) > 96)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.area = int(self.mask.sum())
        self.h, self.w = self.mask.shape
        self.pad = math.ceil(self.clearance * self.size / 448) + 1
        padded = np.pad(self.mask, self.pad)
        # cached distance envelope (no per-pair EDT)
        self.near_mask = ndimage.distance_transform_edt(~padded) < self.clearance * self.size / 448

    def moved(self, x, y):
        """Reuse the tile and distance envelope at a different canvas offset."""
        other = object.__new__(Obj)
        other.__dict__ = dict(self.__dict__, x=int(x), y=int(y))
        return other

    def center(self):
        ys, xs = np.where(self.mask)
        return [round(self.x + float(xs.mean()), 3), round(self.y + float(ys.mean()), 3)]


def pair_intersection(a, b):
    return _intersection(a.mask, a.x, a.y, b.mask, b.x, b.y)


def pair_state(a, b, size=448):
    """Classify using BOTH the relative and scale-correct absolute floor."""
    inter = pair_intersection(a, b)
    if inter:
        minimum = max(OVERLAP_MIN * min(a.area, b.area), max(a.intersection_floor, b.intersection_floor) * (size/448)**2)
        return "overlap" if inter >= minimum else "ambiguous"
    nearby = _intersection(a.near_mask, a.x-a.pad, a.y-a.pad,
                           b.mask, b.x, b.y)
    return "ambiguous" if nearby else "apart"


def _encode_mask(mask):
    """Row-major runs [start, length], for foreground pixels only."""
    indices = np.flatnonzero(mask.ravel())
    if not len(indices):
        return {"encoding": "foreground-runs-row-major", "shape": list(mask.shape), "runs": []}
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    ends = np.r_[starts[1:] - 1, len(indices) - 1]
    return {"encoding": "foreground-runs-row-major", "shape": list(mask.shape),
            "runs": [[int(indices[a]), int(indices[b]-indices[a]+1)] for a, b in zip(starts, ends)]}


def _physical_tile(tile):
    alpha = np.asarray(tile.getchannel("A"))
    ink = alpha > 96
    filled = ndimage.binary_fill_holes(ink)
    labels, count = ndimage.label(filled)
    areas = np.bincount(labels.ravel())[1:]
    if not count or (len(areas) > 1 and sorted(areas)[-2] > max(8, .015*areas.sum())):
        return None
    # Add opaque material only where an enclosed outline leaves transparent
    # pixels. Original RGB/shading is composited without any palette recoloring.
    holes = filled & ~ink
    if holes.any():
        backing = Image.new("RGBA", tile.size, (255, 255, 255, 0))
        backing.putalpha(Image.fromarray((holes.astype(np.uint8)*255)))
        backing.alpha_composite(tile)
        tile = backing
    return tile, filled


class SpriteFactory:
    def __init__(self, rng, size, split, lib, difficulty="hard"):
        if split not in ("train", "eval", "val", "test"):
            raise ValueError("overlap requires an explicit train/eval/val/test split")
        self.rng, self.size, self.split, self.lib = rng, size, split, lib
        self.cfg = TIERS[difficulty]
        self.assets = []
        if lib is not None:
            for class_id in sorted(SOLID_FLUENT_CLASSES.intersection(lib.classes(split=split))):
                self.assets.extend(a for a in lib.variants(class_id, split=split)
                                   if not a.get("counting_notes"))
            if not self.assets:
                raise ValueError(f"no solid Fluent relation assets in split {split!r}")

    def _make(self, target_area):
        rng, size = self.rng, self.size
        span = 64 * size / 448
        family = rng.choices(["fluent", "geometry", "stick"],
                             [0.65 if self.assets else 0, .30, .05])[0]
        angle, scale = rng.uniform(-180, 180), 1.0
        desc = {"family": family, "kind": family, "transform": {
            "angle": round(angle, 5), "size": round(span, 5), "scale": round(scale, 5),
            "mirror": False, "pose": None}}
        if family == "fluent":
            asset = rng.choice(self.assets)
            mirror = rng.random() < .5
            desc["transform"]["mirror"] = mirror
            tile = self.lib.render(asset, size=span, scale=scale, angle=angle,
                                   mirror=mirror)
            desc.update({key: asset[key] for key in
                         ("asset_id", "identity_id", "class_id", "style", "split", "source_name", "path")})
            desc["asset_split"] = asset["split"]
            if asset.get("sha256"):
                desc["asset_sha256"] = asset["sha256"]
            if asset.get("source"):
                desc["asset_source"] = asset["source"]
            physical = _physical_tile(tile)
            if physical is None:
                return None
            tile, mask = physical
            desc["mask_policy"] = "alpha_gt_96_with_enclosed_holes_white_backed"
        else:
            s = max(12, round(span * scale))
            col = rng.choice(PALETTE)
            tile = Image.new("RGBA", (s+8, s+8))
            d = ImageDraw.Draw(tile)
            if family == "stick":
                # Connected capsule; dashed lines would not define one support.
                width = max(7, round(s * rng.uniform(.18, .30)))
                d.rounded_rectangle((4, (s+8-width)/2, s+4, (s+8+width)/2),
                                    radius=width/2, fill=col+(255,))
                desc["style"] = "solid_capsule"
            else:
                shape = rng.choice(("circle", "square", "pentagon", "diamond"))
                desc["shape"] = shape
                if shape == "circle":
                    d.ellipse((4, 4, s+4, s+4), fill=col+(255,))
                elif shape == "square":
                    d.rounded_rectangle((4, 4, s+4, s+4), radius=s*.10, fill=col+(255,))
                else:
                    n = 5 if shape == "pentagon" else 4
                    points = [(4+s/2+s/2*math.cos(-math.pi/2+2*math.pi*i/n),
                               4+s/2+s/2*math.sin(-math.pi/2+2*math.pi*i/n)) for i in range(n)]
                    d.polygon(points, fill=col+(255,))
            tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            mask = np.asarray(tile.getchannel("A")) > 96
            desc["color_rgb"] = list(col)
            desc["mask_policy"] = "alpha_gt_96"
        # Normalize actual physical support area, not nominal PNG canvas size.
        # The same distribution applies to every object before graph labels are
        # assigned. Resize from this original tile to avoid accumulating blur.
        original = tile
        base_area = int(mask.sum())
        if not base_area:
            return None
        factor = math.sqrt(target_area / base_area)
        for _ in range(2):
            tile = original.resize((max(3, round(original.width*factor)),
                                    max(3, round(original.height*factor))),
                                   Image.Resampling.LANCZOS)
            physical = _physical_tile(tile)
            if physical is None:
                return None
            tile, mask = physical
            factor *= math.sqrt(target_area / int(mask.sum()))
        support = np.asarray(tile.getchannel("A")) > 0
        ys, xs = np.where(support)
        if not len(xs):
            return None
        box = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        tile = tile.crop(box)
        mask = mask[box[1]:box[3], box[0]:box[2]]
        minimum_span = (28 if family == "fluent" else 20)*size/448
        if max(tile.size) < minimum_span or max(tile.size) > .43*size:
            return None
        desc["transform"]["scale"] = round(math.sqrt(int(mask.sum())/base_area), 6)
        desc["target_support_area"] = round(target_area, 3)
        desc["size_sampling"] = "shared_scene_budget_population_before_graph_assignment"
        obj = Obj(tile, 0, 0, desc, size, mask, self.cfg["clearance"], self.cfg["intersection"])
        return obj if obj.area >= .95*self.cfg["min_area"]*(size/448)**2 else None

    def make(self, target_area):
        for _ in range(60):
            obj = self._make(target_area)
            if obj is not None:
                return obj
        raise RuntimeError(f"no readable sprite at target support area {target_area:.1f}")


def scene_areas(rng, size, count, difficulty="hard"):
    """An answer-independent ink budget, broadly partitioned before labels.

    Equal pixel areas need not mean equal apparent sizes: silhouettes/aspect
    ratios vary. A wide lognormal distribution adds substantial within-scene
    variation. A shared minimum permits every item to be an overlap endpoint.
    """
    unit = (size/448)**2
    minimum, maximum = TIERS[difficulty]["min_area"]*unit, 11000*unit
    # The same budget distribution applies across all six tiers. Readability
    # floors fit even 64 objects at its lower bound, so no count-conditioned
    # clamp or selective shrinking of dense scenes is needed.
    budget = rng.uniform(.18, .255)*size*size
    if budget <= minimum*count:
        raise ValueError("scene area budget is below the readable support minimum")
    sigma = rng.uniform(.85, 1.55)
    weights = np.array([rng.lognormvariate(0, sigma) for _ in range(count)])
    areas = np.full(count, minimum)
    residual = budget - areas.sum()
    active = np.ones(count, bool)
    while residual > .001:
        additions = residual*weights[active]/weights[active].sum()
        indices = np.flatnonzero(active)
        used = np.minimum(additions, maximum-areas[active])
        areas[indices] += used
        residual -= float(used.sum())
        active &= areas < maximum-.001
        if not active.any():
            break
    rng.shuffle(areas)
    return areas.tolist(), {"requested_support_area": round(budget, 3),
                            "requested_coverage": round(budget/(size*size), 6),
                            "minimum_object_area": minimum,
                            "maximum_object_area": maximum,
                            "lognormal_sigma": round(sigma, 6)}


def gamma_of(objs, size):
    canvas = np.zeros((size, size), bool)
    gamma = [0.0] * len(objs)
    for i in reversed(range(len(objs))):
        obj = objs[i]
        covered = canvas[obj.y:obj.y+obj.h, obj.x:obj.x+obj.w]
        gamma[i] = float(np.count_nonzero(covered & obj.mask)) / obj.area
        covered |= obj.mask
    return gamma


def _groups_for(answer, rng, difficulty):
    """Partition all targets into tier-dependent connected components."""
    result = []
    if difficulty == "ood":
        result.append(("star", 4))  # degree three is evaluation-only
        answer -= 4
    low, high = TIERS[difficulty]["component"]
    count = min(answer//2, max(1, math.ceil(answer / high)))
    # Balance component sizes; no singleton can accidentally become a target.
    base, extra = divmod(answer, count)
    if base < 2:
        raise ValueError("overlap components need at least two vertices")
    result.extend(("chain", base + (i < extra)) for i in range(count))
    rng.shuffle(result)
    return result


def _visually_distinct(a, b):
    if a.desc["family"] == b.desc["family"] == "geometry":
        return math.dist(a.desc["color_rgb"], b.desc["color_rgb"]) >= 110
    if a.desc["family"] == b.desc["family"] == "fluent":
        return (a.desc["class_id"] != b.desc["class_id"]
                and not (a.desc["style"] == b.desc["style"] == "high_contrast"))
    return True


def _attach(rng, size, anchor, candidate, others, band, *, angle_hint=None, back_is_anchor=True):
    """Find a readable contact in the actual occlusion band of the back object."""
    if not _visually_distinct(anchor, candidate):
        return None
    back_area = anchor.area if back_is_anchor else candidate.area
    low = max(band[0]*back_area, OVERLAP_MIN*min(anchor.area, candidate.area),
              anchor.intersection_floor*(size/448)**2)
    high = band[1]*back_area
    if low > high or low > min(anchor.area, candidate.area):
        return None
    span = (math.hypot(anchor.w, anchor.h)+math.hypot(candidate.w, candidate.h))/2
    for trial in range(100):
        angle = rng.uniform(0, 2*math.pi) if angle_hint is None else angle_hint+rng.uniform(-.43, .43)
        dx, dy = math.cos(angle), math.sin(angle)
        # Search a crossing of the desired interval. The binary refinement is
        # local to a detected contact, so disconnected/concave silhouettes do
        # not require a global monotonicity assumption.
        previous = span
        for distance in np.linspace(span, 0, 32):
            x = round(anchor.x+anchor.w/2-candidate.w/2+distance*dx)
            y = round(anchor.y+anchor.h/2-candidate.h/2+distance*dy)
            moved = candidate.moved(x, y)
            inter = pair_intersection(anchor, moved)
            if inter >= low:
                outer, inner = previous, distance
                for _ in range(7):
                    if low <= inter <= high:
                        break
                    mid = (outer+inner)/2
                    moved = candidate.moved(round(anchor.x+anchor.w/2-candidate.w/2+mid*dx),
                                            round(anchor.y+anchor.h/2-candidate.h/2+mid*dy))
                    inter = pair_intersection(anchor, moved)
                    if inter < low:
                        outer = mid
                    else:
                        inner = mid
                if low <= inter <= high and all(pair_state(o, moved, size) == "apart" for o in others):
                    return moved
                break
            previous = distance
    return None


def _local_group(rng, size, population, cfg, topology, require_peak=False):
    # Ordering areas within a preassigned target group allows broad scales
    # without hiding small front objects under larger back ones. Target versus
    # isolated membership itself was assigned independently of the areas.
    population = sorted(population, key=lambda obj: obj.area)
    if topology == "star":
        center = population[-1].moved(size//2-population[-1].w//2, size//2-population[-1].h//2)
        leaves = []
        base_angle = rng.uniform(0, 2*math.pi)
        for i, leaf in enumerate(population[:-1]):
            placed = _attach(rng, size, center, leaf, leaves, (.23, .29),
                             angle_hint=base_angle+2*math.pi*i/3, back_is_anchor=False)
            if placed is None:
                return None
            leaves.append(placed)
        ordered = leaves+[center]
    else:
        first = population[0]
        ordered = [first.moved(size//2-first.w//2, size//2-first.h//2)]
        direction = rng.uniform(0, 2*math.pi)
        for candidate in population[1:]:
            placed = _attach(rng, size, ordered[-1], candidate, ordered[:-1], cfg["edge_band"], angle_hint=direction)
            if placed is None:
                return None
            ordered.append(placed)
    min_x, min_y = min(o.x for o in ordered), min(o.y for o in ordered)
    ordered = [o.moved(o.x-min_x, o.y-min_y) for o in ordered]
    extent = max(max(o.x+o.w, o.y+o.h) for o in ordered)+2
    # Layer order varies independently of size. Interior vertices may sit
    # behind both neighbors, so actual combined occlusion increases while
    # every individual contact stays visually interpretable.
    for _ in range(48):
        measured = gamma_of(ordered, extent)
        if (not require_peak or cfg["gamma_band"][0] <= max(measured)) and max(measured) <= cfg["gamma_band"][1]:
            return ordered
        rng.shuffle(ordered)
    return None


def _place(rng, size, population, answer, cfg, difficulty, factory):
    margin = max(5, round(size*.02))
    groups = _groups_for(answer, rng, difficulty)
    groups.sort(key=lambda item: item[1], reverse=True)
    offset, units = 0, []
    for topology, group_size in groups:
        selected = population[offset:offset+group_size]
        offset += group_size
        group = None
        for group_attempt in range(32):
            if group_attempt and group_attempt % 4 == 0:
                selected = [factory.make(obj.desc["target_support_area"]) for obj in selected]
            group = _local_group(rng, size, selected, cfg, topology, require_peak=not units)
            if group is not None:
                break
        if group is None:
            return None
        for obj in group:
            obj.desc = dict(obj.desc, planned_component=topology, component_size=group_size)
        units.append(group)
    units.extend([[obj] for obj in population[answer:]])
    def extent(unit):
        return max(o.x+o.w for o in unit), max(o.y+o.h for o in unit)
    units.sort(key=lambda unit: math.prod(extent(unit)), reverse=True)
    objs = []
    for unit in units:
        width, height = extent(unit)
        if width >= size-2*margin or height >= size-2*margin:
            return None
        accepted = False
        for _ in range(1800):
            x, y = rng.randint(margin, size-margin-width), rng.randint(margin, size-margin-height)
            shifted = [o.moved(o.x+x, o.y+y) for o in unit]
            if all(pair_state(a, b, size) == "apart" for a in objs for b in shifted):
                objs.extend(shifted)
                accepted = True
                break
        if not accepted:
            return None
    return objs


def _components(neighbors):
    visited, components = set(), []
    for first, ns in enumerate(neighbors):
        if first in visited or not ns:
            continue
        pending, component = [first], []
        while pending:
            vertex = pending.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            component.append(vertex)
            pending.extend(neighbors[vertex])
        components.append(sorted(component))
    return components


def make_overlap_problem(rng, size, difficulty, split, lib, *, answer_target=None):
    """Generate six internal profiles with verified topology and occlusion.

    Number, answer, area population, and thresholds are immutable across local
    retries. All masks and measured relation controls are included for audit.
    """
    if difficulty not in TIERS:
        raise ValueError(f"overlap difficulty must be one of {tuple(TIERS)}")
    if size < 224:
        raise ValueError("overlap size must be at least 224 px")
    cfg = TIERS[difficulty]
    answer_range = OVERLAP_TARGET_RANGES[difficulty]
    answer = rng.randint(*answer_range) if answer_target is None else answer_target
    if not isinstance(answer, int) or not answer_range[0] <= answer <= answer_range[1]:
        raise ValueError(f"answer_target for {difficulty} must be in {answer_range}")
    n = rng.randint(*cfg["n"])
    factory = SpriteFactory(rng, size, split, lib, difficulty)
    areas, area_policy = scene_areas(rng, size, n, difficulty)
    population = [factory.make(area) for area in areas]
    objs = None
    for attempt in range(100):
        if attempt and attempt % 5 == 0:
            population = [factory.make(area) for area in areas]
        rng.shuffle(population)
        objs = _place(rng, size, population, answer, cfg, difficulty, factory)
        if objs is not None and (lib is None or any(o.desc["family"] == "fluent" for o in objs)):
            break
    else:
        raise RuntimeError(f"overlap packing failed after 100 attempts: difficulty={difficulty}, "
                           f"n={n}, answer={answer}, size={size}, controls={cfg}")
    graph = []
    neighbors = [[] for _ in objs]
    for i, a in enumerate(objs):
        for j in range(i+1, len(objs)):
            state = pair_state(a, objs[j], size)
            if state == "ambiguous":
                raise AssertionError("placement admitted an ambiguous pair")
            if state == "overlap":
                inter = pair_intersection(a, objs[j])
                graph.append({"ids": [i, j], "intersection_px": inter,
                              "relative_to_smaller": round(inter/min(a.area, objs[j].area), 6)})
                neighbors[i].append(j)
                neighbors[j].append(i)
    targets = [i for i, ns in enumerate(neighbors) if ns]
    gamma = gamma_of(objs, size)
    occluded = [value for value in gamma if value > 0]
    components = _components(neighbors)
    degree = max(map(len, neighbors))
    if len(targets) != answer or degree > cfg["max_neighbors"]:
        raise AssertionError("mask-derived answer or graph invariant failed")
    if not occluded or max(occluded) < cfg["gamma_band"][0] or max(occluded) > cfg["gamma_band"][1]:
        raise AssertionError("actual occlusion band invariant failed")
    if difficulty == "ood" and degree != 3:
        raise AssertionError("OOD must contain a held-out degree-three branching component")
    image = Image.new("RGB", (size, size), rng.choice(((250, 248, 242), (248, 249, 251), (250, 247, 249))))
    for obj in objs:
        image.paste(obj.tile, (obj.x, obj.y), obj.tile)
    elements = []
    for i, obj in enumerate(objs):
        center = obj.center()
        elements.append({"id": i, **obj.desc, "center": center, "centroid": center,
                         "bbox": [obj.x, obj.y, obj.x+obj.w, obj.y+obj.h],
                         "mask_offset": [obj.x, obj.y], "mask": _encode_mask(obj.mask),
                         "mask_area": obj.area, "occluded_fraction": round(gamma[i], 6),
                         "overlap_neighbors": neighbors[i], "is_target": bool(neighbors[i])})
    controls = {"n_range": list(cfg["n"]), "answer_range": list(answer_range),
                "actual_max_occlusion_band": list(cfg["gamma_band"]),
                "edge_occlusion_band_during_placement": list(cfg["edge_band"]),
                "component_size_range": list(cfg["component"]),
                "max_neighbors": cfg["max_neighbors"],
                "topology": "chains_and_degree_three_star" if difficulty == "ood" else "chains",
                "ood_holdout": "degree_three_star" if difficulty == "ood" else None}
    metrics = {"overlap_targets": len(targets), "overlap_edges": len(graph),
               "component_sizes": sorted(map(len, components)), "max_degree": degree,
               "max_component_size": max(map(len, components)),
               "occluded_object_count": len(occluded),
               "occlusion_min_positive": min(occluded), "occlusion_mean_positive": sum(occluded)/len(occluded),
               "occlusion_max": max(occluded)}
    meta = {"operator": "overlap", "difficulty": difficulty,
            "source": "overlap_tiers+six_tier_relations",
            "source_tier": cfg["source_tier"], "n_objects": n, "n_overlapping": answer,
            "n_pairs_to_judge": n*(n-1)//2, "target_ids": targets, "overlap_graph": graph,
            "gamma_max": round(max(gamma), 6), "gamma_cap": cfg["gamma"],
            "gamma_mean": round(sum(gamma)/n, 6), "clearance_px": cfg["clearance"]*size/448,
            "overlap_min_relative": OVERLAP_MIN, "overlap_min_px": cfg["intersection"]*(size/448)**2,
            "max_neighbors": cfg["max_neighbors"], "split": split, "packing_attempts": attempt+1,
            "difficulty_controls": controls, "difficulty_metrics": metrics,
            "ood_axes": ["heldout_count_range", "degree_three_branching_component", "higher_achieved_occlusion"] if difficulty == "ood" else [],
            "overlap_components": components,
            "families": dict(Counter(o.desc["family"] for o in objs)),
            "styles": dict(Counter(o.desc.get("style", "geometry") for o in objs)),
            "relation_mask_policy": "alpha>96; enclosed transparent interiors receive white backing; conservative solid Fluent allowlist",
            "solution_centroids": [elements[i]["center"] for i in targets],
            "instances": elements, "size_policy": area_policy,
            "total_support_area": sum(o.area for o in objs),
            "size_policy_version": "shared-budget-six-tier"}
    cot = (f"Check whether each object overlaps another object. Count each qualifying object once, "
           f"including every object in a connected group. {answer} of the {n} objects qualify. The answer is {answer}.")
    return image, QUESTION, str(answer), cot, meta, [elements[i] for i in targets]
