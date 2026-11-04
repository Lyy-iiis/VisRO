"""Core board generator for Flow Free.

The question design is a balanced 2 x 2 crossing of scope (one pipe or all
pipes) and statistic (cell length or right-angle turns).  Geometry is sampled
before wording, colour, labels, and annotations.  Two layout recipes are
supported:

* four_pipe: the four-pipe length ladder of length_ladder.py
* diverse_multipipe: a 4/6/8-pipe recipe held to the same length ladder and a
  conservative branch-start turn-count floor

Both recipes use the Backbite Hamiltonian constructor (backbite.py); the
family name describes a layout recipe, not a different path algorithm.
"""
from __future__ import annotations
from visro.seeding import seed_text

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

from . import backbite, board_render, length_ladder, pipe_colors


SIZE = 448
VERSION = "flow_free_board"
# RNG_NAMESPACE is the fixed namespace of every random stream (rng_for); the
# out-of-distribution generator checks it as OOD_COMPAT_VERSION.  Changing it
# changes every board.  ID_NAMESPACE_VERSION only names record ids and images.
RNG_NAMESPACE = seed_text(2511418772087835368491889623307155184469413597376293658418)
OOD_COMPAT_VERSION = RNG_NAMESPACE
ID_NAMESPACE_VERSION = "board"
TIERS = length_ladder.TIERS
TRAIN_TIERS = ("easy", "medium", "hard", "xhard")
OOD_TIER = "xxhard"
ID_DATASET_ROLES = ("train_id", "eval_id", "val_id")
HOLDOUT_ID_ROLES = ("eval_id", "val_id")
DATASET_ROLES = ID_DATASET_ROLES + ("ood_eval",)
ROLE_TO_SPLIT = {
    "train_id": "train",
    "eval_id": "eval",
    "val_id": "val",
    # the xxhard control draws from the eval namespace
    "ood_eval": "eval",
}
ROLE_TO_ID_NAMESPACE = {
    "train_id": "train_id",
    "eval_id": "eval_id",
    "val_id": "val_id",
    "ood_eval": "eval_ood",
}
BASE_SCHEDULE_VERSION = "flow_free_split_schedule"
HOLDOUT_ID_SCHEDULE_VERSION = "flow_free_id_latin_split_schedule"
MAX_ATTEMPTS = 400
MIN_PAIRWISE_COLOR_DE = 40.0

QUESTION_FORMS = (
    "all_pipe_length",
    "one_pipe_length",
    "one_pipe_turn_count",
    "all_pipe_turn_count",
)
FORM_MEANING = {
    "all_pipe_length": ("all", "length"),
    "one_pipe_length": ("one", "length"),
    "one_pipe_turn_count": ("one", "turn_count"),
    "all_pipe_turn_count": ("all", "turn_count"),
}

# This is deliberately capped at eight colours so multi-pipe numbered boards keep
# one-digit labels.  Names are ordinary English colour words used in prompts.
NAMED_PALETTE = tuple(
    c for c in pipe_colors.PIPE_COLORS
    if c[0] in {"red", "blue", "green", "yellow", "orange", "cyan", "pink", "brown"}
)
assert len(NAMED_PALETTE) == 8
_COLOR_BY_NAME = dict(pipe_colors.PIPE_COLORS)
HIGH_CONTRAST_6 = tuple(
    (name, _COLOR_BY_NAME[name])
    for name in ("red", "blue", "green", "yellow", "cyan", "pink")
)
HIGH_CONTRAST_7 = tuple(
    (name, _COLOR_BY_NAME[name])
    for name in ("red", "blue", "green", "yellow", "cyan", "pink", "brown")
)


def color_de_gate(distinct_count: int) -> float:
    """Documented feasible gate for the number of distinct named colours."""
    if distinct_count <= 6:
        return 55.0
    if distinct_count == 7:
        return 50.0
    return MIN_PAIRWISE_COLOR_DE

# The second recipe adds 4/6/8-pipe visual diversity without dropping to easier
# 7/9/11 grids.  Its grids are
# at least as fine as the branch-start length ladder, every tier uses the same
# high-tortuosity setting, and the one-pipe target keeps the exact length-ladder
# length band.  Xhard/xxhard need 17/21 cells respectively: the inherited cut
# routine caps any segment near 2.5x the mean, so an eight-pipe 15x15 board
# cannot contain a 110--135-cell target.
DIVERSE_MULTIPIPE_CFG = {
    "easy":   {"n": 11, "c": 4, "tort": 0.95, "origin": "hard_multipipe"},
    "medium": {"n": 13, "c": 6, "tort": 0.95, "origin": "hard_multipipe"},
    "hard":   {"n": 15, "c": 8, "tort": 0.95, "origin": "hard_multipipe"},
    "xhard":  {"n": 17, "c": 8, "tort": 0.95, "origin": "hard_multipipe"},
    "xxhard": {"n": 21, "c": 8, "tort": 0.95, "origin": "hard_multipipe_ood"},
}

# One-pipe length and turn prompts use the same joint candidate pool.  Length
# support is byte-for-byte the branch-start length ladder.  Turn lower bounds
# are the empirical q10 values of 25k-per-tier samples of that ladder; upper bounds
# are the natural maximum at the longest allowed length.  This retains broad,
# overlapping answers while making the multi-pipe slice slightly harder on
# average rather than silently easier.
DIVERSE_ONE_PIPE_LENGTH_BANDS = dict(length_ladder.LEN_BANDS)
DIVERSE_ONE_PIPE_TURN_BANDS = {
    "easy":   (7, 20),
    "medium": (15, 36),
    "hard":   (23, 53),
    "xhard":  (40, 88),
    "xxhard": (63, 133),
}


@dataclasses.dataclass(frozen=True)
class Spec:
    index: int
    block: int
    question_form: str
    scope: str
    statistic: str
    layout_family: str
    tier: str
    order: str | None
    reference: str | None
    dataset_role: str = "train_id"
    nuisance_row: int = 0
    n_variant: int = 0


@dataclasses.dataclass
class BaseBoard:
    n: int
    segments: list[list[tuple[int, int]]]
    target_index: int
    distractor_index: int
    tortuosity: float
    path_turn_fraction: float
    attempts: int
    base_board_hash: str
    recipe_origin: str


def rng_for(seed: int | str, index: int, stream: str, nonce: int = 0) -> random.Random:
    """Stable independent RNG stream; changing wording cannot perturb geometry."""
    raw = f"{RNG_NAMESPACE}|{seed}|{index}|{stream}|{nonce}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")
    return random.Random(value)


# Every nuisance row is a complete four-record block, so length and turn-count
# receive identical visible factors inside both the all-pipe and one-pipe
# scopes.  The eight rows jointly give the requested 50/25/25 all-pipe order,
# 50/50 one-pipe reference, and exact layout x subtype balance.
NUISANCE_ROWS = (
    ("four_pipe", "pipe_label", "circle"),
    ("diverse_multipipe", "pipe_label", "color_name"),
    ("four_pipe", "ascending", "color_name"),
    ("diverse_multipipe", "ascending", "circle"),
    ("four_pipe", "descending", "circle"),
    ("diverse_multipipe", "descending", "color_name"),
    ("four_pipe", "pipe_label", "color_name"),
    ("diverse_multipipe", "pipe_label", "circle"),
)

# A 128-record train supercycle contains every tier x nuisance-row cell once.
# Its first eight blocks are deliberately balanced too: 100,000 records are
# 781 full supercycles plus exactly this prefix, yielding 25,000 records/tier.
_TRAIN_PREFIX = (
    ("easy", 0), ("medium", 1), ("hard", 6), ("xhard", 7),
    ("easy", 5), ("medium", 4), ("hard", 3), ("xhard", 2),
)
_TRAIN_PREVIEW_SECOND = (
    ("easy", 2), ("medium", 6), ("hard", 1), ("xhard", 0),
    ("easy", 7), ("medium", 3), ("hard", 4), ("xhard", 5),
)
_TRAIN_PREVIEW_PREFIX = _TRAIN_PREFIX + _TRAIN_PREVIEW_SECOND
TRAIN_BLOCK_CYCLE = _TRAIN_PREVIEW_PREFIX + tuple(
    (tier, row)
    for tier in TRAIN_TIERS
    for row in range(len(NUISANCE_ROWS))
    if (tier, row) not in _TRAIN_PREVIEW_PREFIX
)
assert len(TRAIN_BLOCK_CYCLE) == 32


def split_for_role(dataset_role: str) -> str:
    """Return the externally visible split namespace for a dataset role."""
    try:
        return ROLE_TO_SPLIT[dataset_role]
    except KeyError as exc:
        raise ValueError(f"unknown dataset role: {dataset_role!r}") from exc


def id_namespace_for_role(dataset_role: str) -> str:
    """Return the collision-proof record/image namespace for a dataset role."""
    try:
        return ROLE_TO_ID_NAMESPACE[dataset_role]
    except KeyError as exc:
        raise ValueError(f"unknown dataset role: {dataset_role!r}") from exc


def schedule_version_for_role(dataset_role: str) -> str:
    """Return the schedule version string recorded for a dataset role."""
    if dataset_role not in DATASET_ROLES:
        raise ValueError(f"unknown dataset role: {dataset_role!r}")
    return (
        HOLDOUT_ID_SCHEDULE_VERSION
        if dataset_role in HOLDOUT_ID_ROLES
        else BASE_SCHEDULE_VERSION
    )


def schedule(index: int, dataset_role: str = "train_id") -> Spec:
    """Deterministic low-discrepancy assignment of question and nuisance factors.

    ``train_id`` and ``ood_eval`` retain the original four-record paired block
    schedule.  The much smaller ID eval/validation sets use a record-level
    Latin schedule: each consecutive group of four has one item from every
    train tier and one item from every question form.  This makes both margins
    exact for the requested 600-record eval and 100-record validation sets;
    a paired block schedule cannot make a 25-record-per-tier validation set.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if dataset_role not in DATASET_ROLES:
        raise ValueError(f"unknown dataset role: {dataset_role!r}")
    block = index // len(QUESTION_FORMS)

    if dataset_role in HOLDOUT_ID_ROLES:
        # In group g and slot s, tier=s and form=(g+s) mod 4.  Thus every
        # four-record group has both margins exactly once, while successive
        # groups rotate their tier/form pairing.  Over 600 records this gives
        # a 38/37 Latin matrix; over 100 it gives diagonal 7 and
        # off-diagonal 6.
        slot = index % len(TRAIN_TIERS)
        tier = TRAIN_TIERS[slot]
        form = QUESTION_FORMS[(block + slot) % len(QUESTION_FORMS)]
        nuisance_quartet = (block % 32) // 4
        nuisance_row = (nuisance_quartet + slot) % len(NUISANCE_ROWS)
        cycle = block // 32
    else:
        form = QUESTION_FORMS[index % len(QUESTION_FORMS)]
        if dataset_role == "train_id":
            tier, nuisance_row = TRAIN_BLOCK_CYCLE[block % len(TRAIN_BLOCK_CYCLE)]
            cycle = block // len(TRAIN_BLOCK_CYCLE)
        else:
            tier = OOD_TIER
            nuisance_row = block % len(NUISANCE_ROWS)
            cycle = block // len(NUISANCE_ROWS)

    scope, statistic = FORM_MEANING[form]
    layout, all_order, one_reference = NUISANCE_ROWS[nuisance_row]

    # four_pipe easy/medium each have two calibrated grid sizes.  Four four_pipe
    # rows per full train cell split 2/2; medium uses the opposite phase so the
    # unavoidable odd residual in a 100k prefix cancels across the two tiers.
    n_variant = 0
    ns = length_ladder.TIER_NS[tier]
    if layout == "four_pipe" and len(ns) > 1:
        four_pipe_rank = (0, 2, 4, 6).index(nuisance_row)
        phase = 1 if tier == "medium" else 0
        n_variant = (four_pipe_rank + phase + cycle) % len(ns)

    return Spec(
        index=index,
        block=block,
        question_form=form,
        scope=scope,
        statistic=statistic,
        layout_family=layout,
        tier=tier,
        order=all_order if scope == "all" else None,
        reference=one_reference if scope == "one" else None,
        dataset_role=dataset_role,
        nuisance_row=nuisance_row,
        n_variant=n_variant,
    )


def turns(cells: Sequence[Sequence[int]]) -> int:
    """Count internal squares whose incoming and outgoing directions differ."""
    return sum(
        1
        for a, b, c in zip(cells, cells[1:], cells[2:])
        if (b[0] - a[0], b[1] - a[1])
        != (c[0] - b[0], c[1] - b[1])
    )


def transform_cell(cell: tuple[int, int], n: int, rotation: int, reflect: bool):
    r, c = cell
    if reflect:
        c = n - 1 - c
    for _ in range(rotation):
        r, c = c, n - 1 - r
    return r, c


def canonical_board_hash(n: int, segments: Sequence[Sequence[Sequence[int]]]) -> str:
    """Hash geometry modulo segment order/direction and square dihedral transforms."""
    candidates = []
    for reflect in (False, True):
        for rotation in range(4):
            norm_segments = []
            for segment in segments:
                path = tuple(
                    transform_cell((int(cell[0]), int(cell[1])), n, rotation, reflect)
                    for cell in segment
                )
                reverse = tuple(reversed(path))
                norm_segments.append(min(path, reverse))
            candidates.append(tuple(sorted(norm_segments)))
    canonical = min(candidates)
    raw = json.dumps([n, canonical], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_base(n: int, segments: Sequence[Sequence[Sequence[int]]]) -> None:
    if not segments:
        raise ValueError("board has no pipes")
    seen: set[tuple[int, int]] = set()
    for pipe_index, segment in enumerate(segments):
        if len(segment) < 3:
            raise ValueError(f"pipe {pipe_index} has fewer than 3 cells")
        local = [tuple(map(int, cell)) for cell in segment]
        if len(set(local)) != len(local):
            raise ValueError(f"pipe {pipe_index} repeats a cell")
        for a, b in zip(local, local[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                raise ValueError(f"pipe {pipe_index} has a non-adjacent step")
        overlap = seen.intersection(local)
        if overlap:
            raise ValueError(f"pipe {pipe_index} overlaps another pipe at {next(iter(overlap))}")
        seen.update(local)
    want = {(r, c) for r in range(n) for c in range(n)}
    if seen != want:
        raise ValueError(f"segments do not exactly cover {n}x{n}: {len(seen)} cells")


def pipe_adjacency(segments):
    owner = {}
    for i, seg in enumerate(segments):
        for c in seg:
            owner[tuple(c)] = i
    adj = [set() for _ in segments]
    for (r, c), i in owner.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = owner.get((r + dr, c + dc))
            if j is not None and j != i:
                adj[i].add(j); adj[j].add(i)
    return adj


def interleave_depth(a, b):
    """How many cells of `a` sit within one step of a cell of `b`.

    The measure of how tangled the target is with its distractor. Two pipes far
    apart on the board are a trivial judgement however long they are; two that
    weave through each other are the actual difficulty.
    """
    sb = {tuple(c) for c in b}
    n = 0
    for (r, c) in a:
        if any((r + dr, c + dc) in sb
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            n += 1
    return n


def _eligible_pairs(
    segments: Sequence[Sequence[Sequence[int]]], target_candidates: Iterable[int]
) -> list[tuple[int, int]]:
    adjacency = pipe_adjacency(segments)
    pairs = []
    for target in target_candidates:
        for distractor, segment in enumerate(segments):
            if distractor == target or distractor in adjacency[target]:
                continue
            if len(segment) >= 10:
                pairs.append((target, distractor))
    return pairs


def target_calibration_bands(
    spec: Spec,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Return latent-target length/turn gates used by ``sample_base``.

    The four-pipe baseline keeps the length-ladder gate for every scope.
    Multi-pipe all-pipe questions do not have a distinguished queried pipe, so
    they use the complete-board workload instead of a latent target gate.
    Multi-pipe one-pipe questions share one joint gate across both statistics.
    """
    if spec.layout_family == "four_pipe":
        return length_ladder.LEN_BANDS[spec.tier], None
    if spec.layout_family == "diverse_multipipe" and spec.scope == "one":
        return (
            DIVERSE_ONE_PIPE_LENGTH_BANDS[spec.tier],
            DIVERSE_ONE_PIPE_TURN_BANDS[spec.tier],
        )
    return None, None


def target_calibration_profile(spec: Spec) -> str:
    """Name the difficulty contract without overstating all-pipe calibration."""
    if spec.scope == "one" and spec.layout_family == "four_pipe":
        return "target_length_band"
    if spec.scope == "one" and spec.layout_family == "diverse_multipipe":
        return "multipipe_branch_floor_joint_length_turn"
    return "provisional_board_tier"


def eligible_target_pairs(
    segments: Sequence[Sequence[Sequence[int]]], spec: Spec
) -> list[tuple[int, int]]:
    """Return deterministic target/distractor pairs under the spec's gates.

    In particular, this function accepts a whole ``Spec`` but never branches
    on ``spec.statistic``.  The two multi-pipe one-pipe question types therefore
    draw from exactly the same joint length/turn target pool.
    """
    length_band, turn_band = target_calibration_bands(spec)
    lengths = [len(segment) for segment in segments]
    turn_counts = [turns(segment) for segment in segments]
    candidates = []
    for pipe_index, (length, turn_count) in enumerate(zip(lengths, turn_counts)):
        if length_band is not None and not (
            length_band[0] <= length <= length_band[1]
        ):
            continue
        if turn_band is not None and not (
            turn_band[0] <= turn_count <= turn_band[1]
        ):
            continue
        candidates.append(pipe_index)
    return _eligible_pairs(segments, candidates)


def sample_base(
    seed: int | str, index: int, spec: Spec, *, collision_nonce: int = 0
) -> BaseBoard:
    """Sample geometry and a latent target before any question-dependent styling."""
    if spec.layout_family == "four_pipe":
        ns = length_ladder.TIER_NS[spec.tier]
        n = ns[spec.n_variant % len(ns)]
        c = length_ladder.PIPES
        tort = length_ladder.TORT
        origin = "branch_start"
    elif spec.layout_family == "diverse_multipipe":
        cfg = DIVERSE_MULTIPIPE_CFG[spec.tier]
        n, c, tort = cfg["n"], cfg["c"], cfg["tort"]
        origin = str(cfg["origin"])
    else:  # pragma: no cover - guarded by schedule and validation
        raise ValueError(f"unknown layout family: {spec.layout_family}")

    length_band, turn_band = target_calibration_bands(spec)
    for attempt in range(MAX_ATTEMPTS):
        nonce = collision_nonce * MAX_ATTEMPTS + attempt
        geometry_rng = rng_for(seed, index, "geometry", nonce)
        path = backbite.backbite_hamiltonian(n, geometry_rng, tort)
        raw_segments = backbite.cut_into_segments(path, c, geometry_rng)
        segments = [[(int(r), int(col)) for r, col in segment] for segment in raw_segments]
        validate_base(n, segments)
        pairs = eligible_target_pairs(segments, spec)
        if not pairs:
            continue
        target_rng = rng_for(seed, index, "latent_target", nonce)
        target, distractor = pairs[target_rng.randrange(len(pairs))]
        return BaseBoard(
            n=n,
            segments=segments,
            target_index=target,
            distractor_index=distractor,
            tortuosity=tort,
            path_turn_fraction=round(backbite.turn_frac(path), 6),
            attempts=attempt + 1,
            base_board_hash=canonical_board_hash(n, segments),
            recipe_origin=origin,
        )
    raise RuntimeError(
        f"could not sample {spec.layout_family}/{spec.tier}/{spec.scope} "
        f"with target bands length={length_band}, turns={turn_band} "
        f"after {MAX_ATTEMPTS} attempts"
    )


@dataclasses.dataclass
class Labeling:
    segment_to_label: list[int]
    endpoint_sides: list[int]

    @property
    def label_to_segment(self) -> dict[int, int]:
        return {label: i for i, label in enumerate(self.segment_to_label)}


def assign_colors(
    seed: int | str, index: int, spec: Spec, base: BaseBoard
) -> tuple[list[tuple[str, tuple[int, int, int]]], str, float]:
    """Assign a palette while keeping its law identical across statistics."""
    rng = rng_for(seed, index, "palette")
    c = len(base.segments)
    unique_mode = spec.scope == "one" and spec.reference == "color_name"
    distinct_needed = c if unique_mode else c - 1
    if distinct_needed <= len(HIGH_CONTRAST_6):
        palette = list(HIGH_CONTRAST_6)
    elif distinct_needed == len(HIGH_CONTRAST_7):
        palette = list(HIGH_CONTRAST_7)
    elif distinct_needed == len(NAMED_PALETTE):
        palette = list(NAMED_PALETTE)
    else:
        raise ValueError(f"no named palette for {distinct_needed} distinct colours")
    rng.shuffle(palette)
    palette = palette[:distinct_needed]

    if unique_mode:
        colors = palette
        mode = "all_unique_named"
    else:
        # Exactly one non-adjacent pair reuses a colour.  Everyone else is
        # unique, so any repeated-colour ambiguity is intentional and local.
        visible = [color for color in palette if pipe_colors.ring_visible_on(color)]
        candidates = visible if spec.scope == "one" else palette
        shared = candidates[rng.randrange(len(candidates))]
        remaining = [color for color in palette if color != shared]
        rng.shuffle(remaining)
        colors = []
        cursor = 0
        for pipe_index in range(c):
            if pipe_index in (base.target_index, base.distractor_index):
                colors.append(shared)
            else:
                colors.append(remaining[cursor])
                cursor += 1
        mode = "one_nonadjacent_repeated_pair"

    distinct = []
    for color in colors:
        if color not in distinct:
            distinct.append(color)
    distances = [
        pipe_colors.delta_e(a, b)
        for i, a in enumerate(distinct)
        for b in distinct[i + 1:]
    ]
    min_de = round(min(distances), 2) if distances else 0.0
    required_de = color_de_gate(len(distinct))
    if distances and min_de < required_de:
        raise ValueError(
            f"pipe palette falls below dE {required_de}: {min_de}"
        )
    return colors, mode, min_de


def assign_labels(seed: int | str, index: int, base: BaseBoard) -> Labeling:
    """One independently permuted numeral at one random endpoint per pipe."""
    rng = rng_for(seed, index, "pipe_labels")
    labels = list(range(1, len(base.segments) + 1))
    rng.shuffle(labels)
    sides = [rng.randrange(2) for _ in base.segments]
    return Labeling(labels, sides)


def _geometry(n: int, size: int) -> tuple[float, float]:
    margin = int(size * 0.05)
    cell = (size - 2 * margin) / n
    return float(margin), cell


def cell_center(cell_rc: Sequence[int], n: int, size: int) -> tuple[float, float]:
    margin, cell = _geometry(n, size)
    r, c = int(cell_rc[0]), int(cell_rc[1])
    return margin + (c + 0.5) * cell, margin + (r + 0.5) * cell


def draw_ring(
    image: Image.Image,
    cell_rc: Sequence[int],
    n: int,
    size: int,
    *,
    dense_safe: bool = False,
) -> dict[str, Any]:
    """Draw the white target ring without changing base pixels.

    ``dense_safe`` is the main-corpus annotation geometry.  The default is the
    geometry used by the out-of-distribution generator.
    """
    _, cell = _geometry(n, size)
    cx, cy = cell_center(cell_rc, n, size)
    # Stay inside the endpoint square: a 0.47-cell radius plus its dark outer
    # stroke would cross the first shared boundary on dense boards and erase
    # one piece of tracing evidence.
    # The densest main-control board is 21x21 (about 19.2 px/cell).  Keep the
    # complete white+dark annotation comfortably inside 0.5 cell even after
    # Pillow's raster rounding; otherwise the dark outline can replace the
    # first true port pixel on an endpoint that exits toward the ring.
    radius = cell * (0.31 if dense_safe else 0.34)
    stroke = max(4, int(cell * 0.16))
    outer_pad = 1 if dense_safe else 2
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=pipe_colors.RING_RGB,
        width=stroke,
    )
    draw.ellipse(
        [
            cx - radius - outer_pad,
            cy - radius - outer_pad,
            cx + radius + outer_pad,
            cy + radius + outer_pad,
        ],
        outline=(20, 20, 24),
        width=2,
    )
    annotation = {
        "ring_cell": [int(cell_rc[0]), int(cell_rc[1])],
        "ring_radius_px": round(radius, 2),
        "ring_stroke_px": stroke,
    }
    if dense_safe:
        annotation["ring_outer_pad_px"] = outer_pad
    return annotation


def draw_labels(
    image: Image.Image, base: BaseBoard, labeling: Labeling, size: int
) -> list[dict[str, Any]]:
    """Draw high-contrast one-digit badges at independently chosen endpoints."""
    _, cell = _geometry(base.n, size)
    outer_radius = min(16.0, cell * 0.31)
    inner_radius = max(outer_radius - 2.0, 5.0)
    font_size = max(10, min(20, int(round(cell * 0.38))))
    font = load_bundled_font(font_size, "sans-bold")
    draw = ImageDraw.Draw(image)
    annotations = []
    for pipe_index, segment in enumerate(base.segments):
        side = labeling.endpoint_sides[pipe_index]
        cell_rc = segment[0] if side == 0 else segment[-1]
        cx, cy = cell_center(cell_rc, base.n, size)
        outer_box = [
            int(round(cx - outer_radius)), int(round(cy - outer_radius)),
            int(round(cx + outer_radius)), int(round(cy + outer_radius)),
        ]
        inner_box = [
            cx - inner_radius, cy - inner_radius,
            cx + inner_radius, cy + inner_radius,
        ]
        draw.ellipse(outer_box, fill=(16, 17, 20))
        draw.ellipse(inner_box, fill=(248, 246, 236))
        label = labeling.segment_to_label[pipe_index]
        draw.text(
            (cx, cy), str(label), font=font, fill=(18, 20, 24),
            anchor="mm", stroke_width=0,
        )
        annotations.append({
            "pipe_index": pipe_index,
            "label": label,
            "endpoint_side": "start" if side == 0 else "end",
            "cell": [int(cell_rc[0]), int(cell_rc[1])],
            "bbox": outer_box,
            "font_px": font_size,
        })
    return annotations


def _direction(order: str) -> str:
    return "smallest to largest" if order == "ascending" else "largest to smallest"


def question_for(
    seed: int | str,
    index: int,
    spec: Spec,
    base: BaseBoard,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
) -> tuple[str, str]:
    """Return a semantically explicit prompt and stable template identifier."""
    n_pipes = len(base.segments)
    target_color = colors[base.target_index][0]
    if spec.scope == "all" and spec.statistic == "length":
        if spec.order == "pipe_label":
            bank = (
                "Each pipe has one numbered endpoint. Counting both endpoint squares, "
                "how many grid squares do pipes 1 through {n} pass through, respectively? "
                "List the counts in label order (1 to {n}) as comma-separated integers.",
                "The pipes are identified by endpoint numbers 1 through {n}. Include the "
                "two endpoint squares when measuring each pipe. Return their lengths in "
                "numeric label order as comma-separated integers.",
            )
            key = "all_length_pipe_label"
        else:
            bank = (
                "Every pipe has a numbered endpoint. Counting both endpoint squares, find "
                "the length of every pipe and list all {n} lengths from {direction} as "
                "comma-separated integers. The endpoint numbers are identifiers only.",
                "Trace all {n} pipes and count every square each one covers, including its "
                "two endpoints. Sort the resulting counts from {direction} and answer with "
                "comma-separated integers.",
            )
            key = f"all_length_{spec.order}"
    elif spec.scope == "all" and spec.statistic == "turn_count":
        definition = (
            "A turn is an internal square where a pipe enters in one direction and leaves "
            "in a perpendicular direction; endpoint squares never count as turns. "
        )
        if spec.order == "pipe_label":
            bank = (
                "Each pipe has one numbered endpoint. {definition}How many turns do pipes "
                "1 through {n} make, respectively? List the counts in label order "
                "(1 to {n}) as comma-separated integers.",
                "The pipes are identified by endpoint numbers 1 through {n}. {definition}"
                "Return the turn count for each pipe in numeric label order as "
                "comma-separated integers.",
            )
            key = "all_turn_pipe_label"
        else:
            bank = (
                "Every pipe has a numbered endpoint. {definition}Count the turns made by "
                "all {n} pipes, sort the counts from {direction}, and answer with "
                "comma-separated integers. The endpoint numbers are identifiers only.",
                "Trace all {n} pipes. {definition}List all per-pipe turn counts from "
                "{direction} as comma-separated integers.",
            )
            key = f"all_turn_{spec.order}"
    elif spec.scope == "one" and spec.statistic == "length":
        if spec.reference == "circle":
            bank = (
                "One endpoint is marked by a white ring. Follow the pipe starting there. "
                "Counting both endpoint squares, how many grid squares does it pass "
                "through? Answer with a single number.",
                "A white ring identifies one pipe endpoint. How many squares are in that "
                "entire pipe when both endpoint squares are included? Answer with a single "
                "number.",
            )
            key = "one_length_circle"
        else:
            bank = (
                "All pipe colours in this image are unique. Counting both endpoint squares, "
                "how many grid squares does the {color} pipe pass through? Answer with a "
                "single number.",
                "Follow the {color} pipe from one endpoint to the other and include both "
                "endpoint squares in its length. How many squares does it cover? Answer "
                "with a single number.",
            )
            key = "one_length_color"
    else:
        definition = (
            "A turn is an internal square where the pipe enters in one direction and "
            "leaves in a perpendicular direction. Straight-through squares count as zero, "
            "and endpoint squares never count as turns. "
        )
        if spec.reference == "circle":
            bank = (
                "One endpoint is marked by a white ring. Follow the pipe starting there. "
                "{definition}How many turns does this pipe make? Answer with a single "
                "number.",
                "A white ring identifies one pipe endpoint. {definition}Count the turns "
                "along that pipe and answer with a single number.",
            )
            key = "one_turn_circle"
        else:
            bank = (
                "All pipe colours in this image are unique. Follow the {color} pipe. "
                "{definition}How many turns does it make? Answer with a single number.",
                "Trace the {color} pipe from endpoint to endpoint. {definition}Return its "
                "total turn count as a single number.",
            )
            key = "one_turn_color"

    template_rng = rng_for(seed, index, "question_template")
    template_index = template_rng.randrange(len(bank))
    question = bank[template_index].format(
        n=n_pipes,
        color=target_color,
        direction=_direction(spec.order or "ascending"),
        definition=locals().get("definition", ""),
    )
    return question, f"{key}_t{template_index}"


def values_for(base: BaseBoard, statistic: str) -> list[int]:
    if statistic == "length":
        return [len(segment) for segment in base.segments]
    if statistic == "turn_count":
        return [turns(segment) for segment in base.segments]
    raise ValueError(f"unknown statistic: {statistic}")


def answer_for(base: BaseBoard, spec: Spec, labeling: Labeling | None) -> str:
    values = values_for(base, spec.statistic)
    if spec.scope == "one":
        return str(values[base.target_index])
    if spec.order == "pipe_label":
        if labeling is None:
            raise ValueError("pipe-label order requires a labeling")
        order = [labeling.label_to_segment[label] for label in range(1, len(values) + 1)]
        ordered = [values[i] for i in order]
    elif spec.order == "ascending":
        ordered = sorted(values)
    elif spec.order == "descending":
        ordered = sorted(values, reverse=True)
    else:
        raise ValueError(f"unknown all-pipe order: {spec.order}")
    return ",".join(str(value) for value in ordered)


def render_problem(
    seed: int | str,
    index: int,
    spec: Spec,
    base: BaseBoard,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    labeling: Labeling | None,
    appearance: board_render.Appearance,
    size: int,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any] | None]:
    image = board_render.draw_board(base.n, base.segments, colors, size, appearance)
    label_annotations: list[dict[str, Any]] = []
    ring_annotation = None
    if spec.scope == "all":
        if labeling is None:
            raise ValueError("all-pipe images must carry endpoint numbers")
        label_annotations = draw_labels(image, base, labeling, size)
    elif spec.reference == "circle":
        target_cell = base.segments[base.target_index][0]
        ring_annotation = draw_ring(
            image, target_cell, base.n, size, dense_safe=True
        )
    return image, label_annotations, ring_annotation


def port_midpoint_recall(
    image: Image.Image,
    n: int,
    segments: Sequence[Sequence[Sequence[int]]],
    color_rgbs: Sequence[Sequence[int]],
    size: int,
    tolerance: int = 36,
) -> float:
    """Fraction of true cell-to-cell pipe ports visible in the final raster."""
    hits = total = 0
    for segment, rgb in zip(segments, color_rgbs):
        for first, second in zip(segment, segment[1:]):
            ax, ay = cell_center(first, n, size)
            bx, by = cell_center(second, n, size)
            got = image.getpixel(
                (int(round((ax + bx) / 2)), int(round((ay + by) / 2)))
            )
            hits += max(abs(got[channel] - int(rgb[channel])) for channel in range(3)) <= tolerance
            total += 1
    return hits / total if total else 1.0


def cot_for(base: BaseBoard, spec: Spec, answer: str, labeling: Labeling | None) -> str:
    values = values_for(base, spec.statistic)
    noun = "cell lengths" if spec.statistic == "length" else "turn counts"
    if spec.scope == "one":
        target = "the white-ringed pipe" if spec.reference == "circle" else "the named-colour pipe"
        return (
            f"Trace {target}. Its {noun[:-1]} "
            f"is {values[base.target_index]}, so the answer is {answer}."
        )
    if spec.order == "pipe_label" and labeling is not None:
        ordered = [
            values[labeling.label_to_segment[label]]
            for label in range(1, len(values) + 1)
        ]
        return f"The per-pipe {noun} in numeric label order are {ordered}; answer {answer}."
    return f"The per-pipe {noun} are {values}; after {spec.order} sorting, answer {answer}."


def generate_one(
    seed: int,
    index: int,
    size: int = SIZE,
    split: str | None = None,
    dataset_role: str = "train_id",
    *,
    collision_nonce: int = 0,
) -> tuple[dict[str, Any], Image.Image]:
    """Generate one validated record/image pair."""
    if dataset_role not in DATASET_ROLES:
        raise ValueError(f"unknown dataset role: {dataset_role!r}")
    expected_split = split_for_role(dataset_role)
    split = expected_split if split is None else split
    if split != expected_split:
        raise ValueError(
            f"dataset_role={dataset_role!r} requires split={expected_split!r}, "
            f"got {split!r}"
        )
    # Namespace every stochastic stream by split.  Reusing a seed/index plan
    # for train and eval therefore cannot silently produce the same pixels.
    rng_seed = f"{split}|{seed}"
    spec = schedule(index, dataset_role)
    base = sample_base(rng_seed, index, spec, collision_nonce=collision_nonce)
    colors, palette_mode, min_color_de = assign_colors(rng_seed, index, spec, base)
    appearance = board_render.appearance_for(rng_seed, spec)
    labeling = assign_labels(rng_seed, index, base) if spec.scope == "all" else None
    image, label_annotations, ring_annotation = render_problem(
        rng_seed, index, spec, base, colors, labeling, appearance, size
    )
    answer = answer_for(base, spec, labeling)
    question, template_id = question_for(rng_seed, index, spec, base, colors)

    lengths = [len(segment) for segment in base.segments]
    turn_counts = [turns(segment) for segment in base.segments]
    segment_rows = []
    for pipe_index, (segment, color) in enumerate(zip(base.segments, colors)):
        label = labeling.segment_to_label[pipe_index] if labeling else None
        label_row = next(
            (row for row in label_annotations if row["pipe_index"] == pipe_index), None
        )
        segment_rows.append({
            "pipe_id": f"pipe{pipe_index}",
            "pipe_index": pipe_index,
            "cells": [[r, c] for r, c in segment],
            "length": lengths[pipe_index],
            "turn_count": turn_counts[pipe_index],
            "color_name": color[0],
            "color_rgb": list(color[1]),
            "display_label": label,
            "label_cell": label_row["cell"] if label_row else None,
            "label_endpoint_side": label_row["endpoint_side"] if label_row else None,
            "label_bbox": label_row["bbox"] if label_row else None,
        })

    target_color = colors[base.target_index]
    distractor_color = colors[base.distractor_index]
    appearance_meta = board_render.metadata(appearance, colors, pipe_colors.lab)
    appearance_meta["final_port_midpoint_recall"] = round(
        port_midpoint_recall(
            image, base.n, base.segments, [rgb for _, rgb in colors], size
        ),
        6,
    )
    target_length_band, target_turn_band = target_calibration_bands(spec)
    calibration = target_calibration_profile(spec)
    target_length = lengths[base.target_index]
    target_turn_count = turn_counts[base.target_index]
    target_turn_density = round(target_turn_count / (target_length - 2), 6)
    target_pairs = eligible_target_pairs(base.segments, spec)
    eligible_target_count = len({target for target, _ in target_pairs})
    image_sha = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    meta: dict[str, Any] = {
        "schema_version": VERSION,
        "rng_namespace_version": RNG_NAMESPACE,
        "id_namespace_version": ID_NAMESPACE_VERSION,
        "generation_seed": seed,
        "rng_split_namespace": split,
        "dataset_role": dataset_role,
        "is_ood": dataset_role == "ood_eval",
        "allowed_tiers": list(
            TRAIN_TIERS if dataset_role in ID_DATASET_ROLES else (OOD_TIER,)
        ),
        "schedule_version": schedule_version_for_role(dataset_role),
        "nuisance_row": spec.nuisance_row,
        "n_variant": spec.n_variant,
        "global_index": index,
        "image_size": size,
        "scope": spec.scope,
        "statistic": spec.statistic,
        "order": spec.order,
        "reference": spec.reference,
        "question_form": spec.question_form,
        "difficulty": spec.tier,
        "tier": spec.tier,
        "layout_family": spec.layout_family,
        "path_program": "backbite_hamiltonian",
        "recipe_origin": base.recipe_origin,
        "difficulty_calibration": calibration,
        "target_length_calibration_band": (
            list(target_length_band) if target_length_band is not None else None
        ),
        "target_turn_count_calibration_band": (
            list(target_turn_band) if target_turn_band is not None else None
        ),
        "target_length_calibration_passed": (
            target_length_band[0] <= target_length <= target_length_band[1]
            if target_length_band is not None else None
        ),
        "target_turn_count_calibration_passed": (
            target_turn_band[0] <= target_turn_count <= target_turn_band[1]
            if target_turn_band is not None else None
        ),
        "target_length": target_length,
        "target_turn_count": target_turn_count,
        "target_turn_density": target_turn_density,
        "eligible_target_count": eligible_target_count,
        "eligible_target_pair_count": len(target_pairs),
        "grid_n": base.n,
        "num_pipes": len(base.segments),
        "num_colors": len(set(color[0] for color in colors)),
        "tortuosity": base.tortuosity,
        "path_turn_fraction": base.path_turn_fraction,
        "generation_attempts": base.attempts,
        "base_board_hash": base.base_board_hash,
        "image_pixel_sha256": image_sha,
        "palette_mode": palette_mode,
        "min_pairwise_color_de": min_color_de,
        "pairwise_color_de_gate": color_de_gate(len(set(color[0] for color in colors))),
        "all_lengths": lengths,
        "all_turns": turn_counts,
        "target_index": base.target_index,
        "target_pipe_id": f"pipe{base.target_index}",
        "target_color": target_color[0],
        "distractor_index": base.distractor_index,
        "distractor_pipe_id": f"pipe{base.distractor_index}",
        "distractor_color": distractor_color[0],
        "interleave_depth": interleave_depth(
            base.segments[base.target_index], base.segments[base.distractor_index]
        ),
        "segments": segment_rows,
        "labels_present": spec.scope == "all",
        "label_to_segment": (
            {str(k): v for k, v in sorted(labeling.label_to_segment.items())}
            if labeling else {}
        ),
        "label_annotations": label_annotations,
        "ring_present": ring_annotation is not None,
        "ring": ring_annotation,
        "template_id": template_id,
        "factor_block": spec.block,
        "collision_nonce": collision_nonce,
        "split": split,
        "solution_method_note": (
            "intentional_color_area_shortcut_for_named_pipe_length"
            if spec.scope == "one"
            and spec.statistic == "length"
            and spec.reference == "color_name"
            else None
        ),
        "design_note": (
            "All-pipe images always carry randomized endpoint numbers, including "
            "ascending and descending prompts, so label presence cannot reveal answer order."
        ),
        **appearance_meta,
    }
    split_namespace = id_namespace_for_role(dataset_role)
    record_id = f"flow_free_{ID_NAMESPACE_VERSION}_{split_namespace}_{index:08d}"
    record = {
        "id": record_id,
        "source_id": record_id,
        "task": "flow_free",
        "task_type": "visual_tracking",
        "subtype": "lines_observation",
        "level": "fine",
        "difficulty": spec.tier,
        "tier": spec.tier,
        "question_form": spec.question_form,
        "scope": spec.scope,
        "statistic": spec.statistic,
        "layout_family": spec.layout_family,
        "order": spec.order,
        "reference": spec.reference,
        "question": question,
        "ans_type": "blank",
        "options": [],
        "answer": answer,
        "cot": cot_for(base, spec, answer, labeling),
        "image": (
            f"images/{ID_NAMESPACE_VERSION}_{split_namespace}_"
            f"{index:08d}_flow.png"
        ),
        "meta": meta,
    }
    validate_record(record, image)
    return record, image


def rectangles_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def validate_record(record: dict[str, Any], image: Image.Image | None = None) -> None:
    """Independent semantic and basic raster contract for a generated record."""
    meta = record["meta"]
    role = meta.get("dataset_role")
    if role not in DATASET_ROLES:
        raise ValueError(f"unknown dataset_role {role!r}")
    if role in ID_DATASET_ROLES and record.get("tier") not in TRAIN_TIERS:
        raise ValueError(f"{role} record contains an OOD tier")
    if role == "ood_eval" and record.get("tier") != OOD_TIER:
        raise ValueError("ood_eval record is not xxhard")
    if meta.get("schema_version") != VERSION:
        raise ValueError("schema version mismatch")
    if meta.get("rng_namespace_version") != RNG_NAMESPACE:
        raise ValueError("RNG namespace version mismatch")
    if meta.get("id_namespace_version") != ID_NAMESPACE_VERSION:
        raise ValueError("ID namespace version mismatch")
    if meta.get("schedule_version") != schedule_version_for_role(role):
        raise ValueError("schedule version disagrees with dataset role")
    calibration_fields = (
        "difficulty_calibration",
        "target_length_calibration_band",
        "target_turn_count_calibration_band",
        "target_length_calibration_passed",
        "target_turn_count_calibration_passed",
        "target_length",
        "target_turn_count",
        "target_turn_density",
        "eligible_target_count",
        "eligible_target_pair_count",
    )
    missing_calibration_fields = [
        field for field in calibration_fields if field not in meta
    ]
    if missing_calibration_fields:
        raise ValueError(
            "missing calibration metadata: "
            + ", ".join(missing_calibration_fields)
        )
    global_index = int(meta.get("global_index", -1))
    if global_index < 0:
        raise ValueError("invalid global index")
    expected_spec = schedule(global_index, role)
    for field in (
        "difficulty", "tier", "question_form", "scope", "statistic",
        "layout_family", "order", "reference",
    ):
        if record.get(field) != meta.get(field):
            raise ValueError(f"top-level/meta field mismatch: {field}")
        if record.get(field) != getattr(
            expected_spec, "tier" if field == "difficulty" else field
        ):
            raise ValueError(f"record field disagrees with schedule: {field}")
    if int(meta.get("factor_block", -1)) != expected_spec.block:
        raise ValueError("factor block disagrees with schedule")
    if int(meta.get("nuisance_row", -1)) != expected_spec.nuisance_row:
        raise ValueError("nuisance row disagrees with schedule")
    if int(meta.get("n_variant", -1)) != expected_spec.n_variant:
        raise ValueError("grid-size variant disagrees with schedule")
    expected_split = split_for_role(role)
    if meta.get("split") != expected_split:
        raise ValueError("dataset_role and split disagree")
    if "split" in record and record.get("split") != expected_split:
        raise ValueError("dataset_role and top-level split disagree")
    if meta.get("rng_split_namespace") != expected_split:
        raise ValueError("dataset_role and RNG split namespace disagree")
    expected_appearance = board_render.appearance_for(
        f"{expected_split}|{meta.get('generation_seed')}", expected_spec
    )
    for field, expected in (
        ("appearance_family", expected_appearance.family),
        ("background_rgb", list(expected_appearance.background_rgb)),
        ("grid_rgb", list(expected_appearance.grid_rgb)),
        ("texture_style", expected_appearance.texture_style),
        ("texture_amplitude_rgb", expected_appearance.texture_amplitude),
        ("texture_scale_px", expected_appearance.texture_scale_px),
        ("texture_seed", expected_appearance.texture_seed),
        ("rgb_jitter_bound", board_render.RGB_JITTER),
    ):
        if meta.get(field) != expected:
            raise ValueError(f"appearance metadata disagrees with schedule: {field}")
    if bool(meta.get("is_ood")) != (role == "ood_eval"):
        raise ValueError("dataset_role and is_ood disagree")
    expected_tiers = list(TRAIN_TIERS if role in ID_DATASET_ROLES else (OOD_TIER,))
    if meta.get("allowed_tiers") != expected_tiers:
        raise ValueError("dataset_role and allowed_tiers disagree")
    expected_namespace = id_namespace_for_role(role)
    expected_id = (
        f"flow_free_{ID_NAMESPACE_VERSION}_{expected_namespace}_{global_index:08d}"
    )
    if record.get("id") != expected_id or record.get("source_id") != expected_id:
        raise ValueError("dataset_role and record id namespace disagree")
    expected_image = (
        f"images/{ID_NAMESPACE_VERSION}_{expected_namespace}_"
        f"{global_index:08d}_flow.png"
    )
    if record.get("image") != expected_image:
        raise ValueError("dataset_role and image namespace disagree")
    if float(meta.get("min_pipe_background_de", 0.0)) < 40.0:
        raise ValueError("pipe/background contrast is below the safety gate")
    if float(meta.get("min_pipe_grid_de", 0.0)) < 30.0:
        raise ValueError("pipe/grid contrast is below the safety gate")
    expected_color_gate = color_de_gate(int(meta.get("num_colors", 0)))
    if float(meta.get("pairwise_color_de_gate", -1.0)) != expected_color_gate:
        raise ValueError("pipe-colour dE gate metadata disagrees")
    if float(meta.get("min_pairwise_color_de", 0.0)) < expected_color_gate:
        raise ValueError("distinct pipe colours are below the safety gate")
    if meta.get("pipe_style") != "orthogonal_solid":
        raise ValueError("standard records require orthogonal solid pipes")
    n = int(meta["grid_n"])
    rows = meta["segments"]
    segments = [[tuple(cell) for cell in row["cells"]] for row in rows]
    validate_base(n, segments)
    if len(rows) != int(meta["num_pipes"]):
        raise ValueError("num_pipes disagrees with segments")
    lengths = [len(segment) for segment in segments]
    turn_counts = [turns(segment) for segment in segments]
    if lengths != meta["all_lengths"] or turn_counts != meta["all_turns"]:
        raise ValueError("stored length/turn vectors disagree with segment geometry")
    for pipe_index, (row, segment, length, turn_count) in enumerate(
        zip(rows, segments, lengths, turn_counts)
    ):
        if row.get("pipe_index") != pipe_index or row.get("pipe_id") != f"pipe{pipe_index}":
            raise ValueError("segment row identity disagrees")
        if row.get("length") != length or row.get("turn_count") != turn_count:
            raise ValueError("segment row statistic disagrees")
        expected_rgb = _COLOR_BY_NAME.get(row.get("color_name"))
        if expected_rgb is None or list(expected_rgb) != row.get("color_rgb"):
            raise ValueError("segment row colour name/RGB disagrees")
    background_rgb = tuple(meta["background_rgb"])
    grid_rgb = tuple(meta["grid_rgb"])
    pipe_rgbs = [tuple(row["color_rgb"]) for row in rows]
    measured_pipe_background = round(min(
        math.dist(pipe_colors.lab(background_rgb), pipe_colors.lab(rgb)) for rgb in pipe_rgbs
    ), 2)
    measured_pipe_grid = round(min(
        math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(rgb)) for rgb in pipe_rgbs
    ), 2)
    measured_grid_background = round(
        math.dist(pipe_colors.lab(background_rgb), pipe_colors.lab(grid_rgb)), 2
    )
    for field, measured in (
        ("min_pipe_background_de", measured_pipe_background),
        ("min_pipe_grid_de", measured_pipe_grid),
        ("grid_background_de", measured_grid_background),
    ):
        if float(meta.get(field, -1.0)) != measured:
            raise ValueError(f"stored appearance contrast disagrees: {field}")
    if measured_pipe_background < 40.0 or measured_pipe_grid < 30.0:
        raise ValueError("recomputed pipe/surface contrast is below the safety gate")
    if measured_grid_background < 18.0:
        raise ValueError("recomputed grid/background contrast is below the safety gate")
    if sum(lengths) != n * n:
        raise ValueError("pipe lengths do not sum to board area")
    if any(not 0 <= value <= length - 2 for value, length in zip(turn_counts, lengths)):
        raise ValueError("turn count outside geometric bounds")
    if canonical_board_hash(n, segments) != meta["base_board_hash"]:
        raise ValueError("base_board_hash disagrees with geometry")
    full_path = [cell for segment in segments for cell in segment]
    if any(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
        for first, second in zip(full_path, full_path[1:])
    ):
        raise ValueError("segment ordering no longer forms the source Hamiltonian path")
    if round(backbite.turn_frac(full_path), 6) != float(meta["path_turn_fraction"]):
        raise ValueError("path turn fraction disagrees with geometry")

    scope = record["scope"]
    statistic = record["statistic"]
    values = lengths if statistic == "length" else turn_counts
    target = int(meta["target_index"])
    distractor = int(meta["distractor_index"])
    if not 0 <= target < len(segments) or not 0 <= distractor < len(segments):
        raise ValueError("target/distractor index is outside the segment vector")
    expected_length_band, expected_turn_band = target_calibration_bands(expected_spec)
    expected_profile = target_calibration_profile(expected_spec)
    if meta.get("difficulty_calibration") != expected_profile:
        raise ValueError("difficulty calibration profile disagrees with schedule")
    for field, band in (
        ("target_length_calibration_band", expected_length_band),
        ("target_turn_count_calibration_band", expected_turn_band),
    ):
        expected_band = list(band) if band is not None else None
        if meta.get(field) != expected_band:
            raise ValueError(f"{field} disagrees with schedule")
    target_length = lengths[target]
    target_turn_count = turn_counts[target]
    if int(meta.get("target_length", -1)) != target_length:
        raise ValueError("target_length disagrees with geometry")
    if int(meta.get("target_turn_count", -1)) != target_turn_count:
        raise ValueError("target_turn_count disagrees with geometry")
    expected_turn_density = round(target_turn_count / (target_length - 2), 6)
    if float(meta.get("target_turn_density", -1.0)) != expected_turn_density:
        raise ValueError("target_turn_density disagrees with geometry")
    expected_length_pass = (
        expected_length_band[0] <= target_length <= expected_length_band[1]
        if expected_length_band is not None else None
    )
    expected_turn_pass = (
        expected_turn_band[0] <= target_turn_count <= expected_turn_band[1]
        if expected_turn_band is not None else None
    )
    if meta["target_length_calibration_passed"] is not expected_length_pass:
        raise ValueError("target length calibration result disagrees with geometry")
    if meta["target_turn_count_calibration_passed"] is not expected_turn_pass:
        raise ValueError("target turn calibration result disagrees with geometry")
    if expected_length_band is not None and not expected_length_pass:
        raise ValueError("target length is outside its calibration band")
    if expected_turn_band is not None and not expected_turn_pass:
        raise ValueError("target turn count is outside its calibration band")
    expected_pairs = eligible_target_pairs(segments, expected_spec)
    if (target, distractor) not in expected_pairs:
        raise ValueError("target/distractor pair is outside the eligible pool")
    if int(meta.get("eligible_target_count", -1)) != len(
        {candidate for candidate, _ in expected_pairs}
    ):
        raise ValueError("eligible_target_count disagrees with geometry")
    if int(meta.get("eligible_target_pair_count", -1)) != len(expected_pairs):
        raise ValueError("eligible_target_pair_count disagrees with geometry")
    colors = [row["color_name"] for row in rows]
    if len(set(colors)) != int(meta["num_colors"]):
        raise ValueError("num_colors disagrees with segment rows")
    distinct_rgbs = []
    for row in rows:
        rgb = tuple(row["color_rgb"])
        if rgb not in distinct_rgbs:
            distinct_rgbs.append(rgb)
    distances = [
        math.dist(pipe_colors.lab(first), pipe_colors.lab(second))
        for index, first in enumerate(distinct_rgbs)
        for second in distinct_rgbs[index + 1:]
    ]
    measured_color_de = round(min(distances), 2) if distances else 0.0
    if measured_color_de != float(meta["min_pairwise_color_de"]):
        raise ValueError("min_pairwise_color_de disagrees with segment rows")
    if meta.get("target_pipe_id") != f"pipe{target}" or meta.get("target_color") != colors[target]:
        raise ValueError("target metadata disagrees with segment rows")
    if (
        meta.get("distractor_pipe_id") != f"pipe{distractor}"
        or meta.get("distractor_color") != colors[distractor]
    ):
        raise ValueError("distractor metadata disagrees with segment rows")
    labeling = None
    if scope == "one":
        expected = str(values[target])
        if record["reference"] == "color_name":
            if len(set(colors)) != len(colors):
                raise ValueError("color-name target requires globally unique pipe colors")
            if meta["ring_present"]:
                raise ValueError("color-name target must not carry a white ring")
            if meta["target_color"] not in record["question"]:
                raise ValueError("target color name is absent from the question")
        elif record["reference"] == "circle":
            if colors[target] != colors[distractor]:
                raise ValueError("circle target and distractor must share a color")
            if not meta["ring_present"]:
                raise ValueError("circle target is missing its ring")
        else:
            raise ValueError("one-pipe record has an unknown reference mode")
    else:
        if not meta["labels_present"]:
            raise ValueError("every all-pipe image must carry randomized endpoint labels")
        label_map = {int(k): int(v) for k, v in meta["label_to_segment"].items()}
        if set(label_map) != set(range(1, len(segments) + 1)):
            raise ValueError("endpoint labels are not a bijection over 1..num_pipes")
        if set(label_map.values()) != set(range(len(segments))):
            raise ValueError("label_to_segment is not a segment bijection")
        segment_to_label = [0] * len(segments)
        for label, segment_index in label_map.items():
            segment_to_label[segment_index] = label
        sides = [0 if row["label_endpoint_side"] == "start" else 1 for row in rows]
        labeling = Labeling(segment_to_label, sides)
        order = record["order"]
        if order == "pipe_label":
            expected_values = [values[label_map[label]] for label in sorted(label_map)]
        elif order == "ascending":
            expected_values = sorted(values)
        elif order == "descending":
            expected_values = sorted(values, reverse=True)
        else:
            raise ValueError("all-pipe record has an unknown answer order")
        expected = ",".join(str(value) for value in expected_values)
        boxes = [row["label_bbox"] for row in rows]
        if any(box is None for box in boxes):
            raise ValueError("an all-pipe segment is missing its label bbox")
        canvas_size = int(meta["image_size"])
        for i, box in enumerate(boxes):
            if not (0 <= box[0] < box[2] < canvas_size and 0 <= box[1] < box[3] < canvas_size):
                raise ValueError(f"label {i} is outside the canvas")
            for other in boxes[i + 1:]:
                if rectangles_overlap(box, other):
                    raise ValueError("number-label badges overlap")
        for row, segment in zip(rows, segments):
            if tuple(row["label_cell"]) not in (segment[0], segment[-1]):
                raise ValueError("a number label is not placed on a pipe endpoint")
            expected_label = next(
                label for label, segment_index in label_map.items()
                if segment_index == row["pipe_index"]
            )
            if row.get("display_label") != expected_label:
                raise ValueError("segment display label disagrees with label_to_segment")
            annotation = next(
                (item for item in meta["label_annotations"]
                 if item.get("pipe_index") == row["pipe_index"]),
                None,
            )
            if annotation is None or any(
                annotation.get(key) != row.get(row_key)
                for key, row_key in (
                    ("label", "display_label"), ("cell", "label_cell"),
                    ("endpoint_side", "label_endpoint_side"), ("bbox", "label_bbox"),
                )
            ):
                raise ValueError("label annotation disagrees with segment row")

    if expected != str(record["answer"]):
        raise ValueError(f"answer {record['answer']!r} != recomputed {expected!r}")
    if " " in str(record["answer"]):
        raise ValueError("answer must use canonical comma-separated formatting without spaces")

    validation_base = BaseBoard(
        n=n, segments=segments, target_index=target, distractor_index=distractor,
        tortuosity=float(meta["tortuosity"]),
        path_turn_fraction=float(meta["path_turn_fraction"]), attempts=1,
        base_board_hash=meta["base_board_hash"], recipe_origin=str(meta["recipe_origin"]),
    )
    colors_with_rgb = [
        (row["color_name"], tuple(row["color_rgb"])) for row in rows
    ]
    rng_seed = f"{expected_split}|{meta['generation_seed']}"
    expected_question, expected_template = question_for(
        rng_seed, global_index, expected_spec, validation_base, colors_with_rgb
    )
    if record.get("question") != expected_question or meta.get("template_id") != expected_template:
        raise ValueError("question/template disagrees with deterministic generation")
    if record.get("cot") != cot_for(validation_base, expected_spec, expected, labeling):
        raise ValueError("chain-of-thought summary disagrees with geometry")

    # Every repeated colour pair must remain separate in the underlying cell
    # graph.  This catches visually fused same-colour pipes before raster QA.
    adjacency = pipe_adjacency(segments)
    for i, color in enumerate(colors):
        for j in range(i + 1, len(colors)):
            if color == colors[j] and j in adjacency[i]:
                raise ValueError("same-colour pipes touch and would be ambiguous")

    if image is not None:
        if image.size != (int(meta["image_size"]), int(meta["image_size"])):
            raise ValueError("image dimensions disagree with metadata")
        if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != meta["image_pixel_sha256"]:
            raise ValueError("image hash disagrees with rendered pixels")
        measured_recall = round(
            port_midpoint_recall(
                image, n, segments, [row["color_rgb"] for row in rows], image.width
            ),
            6,
        )
        if measured_recall != float(meta.get("final_port_midpoint_recall", -1.0)):
            raise ValueError("final port recall metadata disagrees")
        if measured_recall != 1.0:
            raise ValueError("final image lost a true pipe port")
        if scope == "one" and record["reference"] == "circle":
            cell_rc = meta["ring"]["ring_cell"]
            cx, cy = cell_center(cell_rc, n, image.width)
            pixel = image.getpixel((int(round(cx)), int(round(cy))))
            target_rgb = tuple(rows[target]["color_rgb"])
            if max(abs(a - b) for a, b in zip(pixel, target_rgb)) > 60:
                raise ValueError("ring center does not lie on the target pipe")
