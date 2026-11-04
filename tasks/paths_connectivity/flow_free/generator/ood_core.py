"""Flow Free xxhard out-of-distribution boards and paired ablations.

This module deliberately stays separate from the ID train/eval generator.  It
crosses two shifts in four arms:

* length_dose_control: Backbite geometry + ID renderer
* program_only: Prim macro-cycle geometry + ID renderer
* appearance_only: the exact control board + OOD renderer
* combined: the exact alternative-program board + OOD renderer

The alternative program is constructive and has no search fallback.  A random
Prim spanning tree on a 7x7 macro-grid splices 2x2 cell cycles into one 14x14
Hamiltonian cycle; opening that cycle and attaching the unused outer row and
column creates a 15x15 Hamiltonian path.  Random D4 transforms remove a fixed
annex side while keeping the program signature checkable.
"""
from __future__ import annotations
from visro.seeding import seed_text

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Sequence

from PIL import Image, ImageDraw

from . import backbite, board_core, board_render, length_ladder, pipe_colors

RNG_NAMESPACE = seed_text(13040031800777061829964418727092606252312879866198315122769559573253029987335780378796711730)  # fixed RNG namespace (rng_for)
SIZE = 448
N = 15
PIPES = 4
TARGET_BAND = (110, 135)
ARMS = (
    "length_dose_control",
    "program_only",
    "appearance_only",
    "combined",
)
PROGRAM_ARMS = {"program_only", "combined"}
APPEARANCE_ARMS = {"appearance_only", "combined"}
ORDER_BY_REP = ("pipe_label", "pipe_label", "ascending", "descending")
REFERENCE_BY_REP = ("circle", "circle", "color_name", "color_name")
OOD_COMPONENT_STYLES = ("rounded_corner", "port_locked_sine")
OOD_PIPE_STYLE = "mixed_rounded_sine_port_locked"

OOD_SURFACE_FAMILIES = (
    (
        "dark_patchwork",
        (
            (22, 31, 44),
            (40, 28, 43),
            (24, 43, 38),
            (43, 46, 55),
            (30, 35, 50),
            (42, 35, 48),
        ),
        (
            (144, 144, 144),
            (184, 192, 152),
            (160, 176, 120),
            (176, 192, 136),
        ),
        (176, 160, 186),
    ),
    (
        "light_patchwork",
        (
            (240, 224, 192),
            (216, 192, 232),
            (192, 208, 144),
            (224, 192, 208),
            (232, 224, 216),
            (248, 224, 168),
        ),
        (
            (56, 56, 64),
            (64, 64, 64),
            (64, 64, 96),
            (64, 64, 120),
        ),
        (112, 80, 160),
    ),
)
# The table balances dark/light within pipe-label, circle-reference, and
# colour-reference cases.  Length and turn-count cases sharing a nuisance key
# always use the same surface family.
OOD_SURFACE_BY_PAIR = (0, 1, 1, 0, 1, 0, 0, 1)
PICTOGRAMS = ("leaf", "fish", "cloud", "butterfly", "apple", "umbrella")


def ood_rng_for(
    seed: int | str, index: int, stream: str, nonce: int = 0
) -> random.Random:
    """OOD-local stable RNG whose namespace is explicit in OOD metadata."""
    raw = f"{RNG_NAMESPACE}|{seed}|{index}|{stream}|{nonce}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:16], "big"))


@dataclasses.dataclass
class SemanticPair:
    spec: board_core.Spec
    control: board_core.BaseBoard
    program: board_core.BaseBoard
    control_program_meta: dict[str, Any]
    program_meta: dict[str, Any]
    attempts: int


def _edge(a: tuple[int, int], b: tuple[int, int]):
    return tuple(sorted((a, b)))


def _prim_tree(m: int, rng: random.Random):
    root = (rng.randrange(m), rng.randrange(m))
    seen = {root}
    frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
    tree = []

    def add(cell: tuple[int, int]) -> None:
        row, column = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = row + dr, column + dc
            if 0 <= other[0] < m and 0 <= other[1] < m and other not in seen:
                frontier.append((cell, other))

    add(root)
    while frontier:
        index = rng.randrange(len(frontier))
        first, second = frontier.pop(index)
        if second in seen:
            continue
        seen.add(second)
        tree.append((first, second))
        add(second)
    if len(tree) != m * m - 1:
        raise AssertionError("Prim tree did not span the macro-grid")
    return root, tree


def prim_macrocycle_odd_annex(rng: random.Random) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Construct a randomized 15x15 Hamiltonian path without Backbite."""
    macro_n = 7
    edges = set()
    for row in range(macro_n):
        for column in range(macro_n):
            tl = (2 * row, 2 * column)
            tr = (2 * row, 2 * column + 1)
            br = (2 * row + 1, 2 * column + 1)
            bl = (2 * row + 1, 2 * column)
            edges.update((_edge(tl, tr), _edge(tr, br), _edge(br, bl), _edge(bl, tl)))

    root, tree = _prim_tree(macro_n, rng)
    for first, second in tree:
        row, column = first
        other_row, other_column = second
        if row == other_row:
            if column > other_column:
                column, other_column = other_column, column
            left_top = (2 * row, 2 * column + 1)
            left_bottom = (2 * row + 1, 2 * column + 1)
            right_top = (2 * row, 2 * column + 2)
            right_bottom = (2 * row + 1, 2 * column + 2)
            edges.remove(_edge(left_top, left_bottom))
            edges.remove(_edge(right_top, right_bottom))
            edges.add(_edge(left_top, right_top))
            edges.add(_edge(left_bottom, right_bottom))
        else:
            if row > other_row:
                row, other_row = other_row, row
            upper_left = (2 * row + 1, 2 * column)
            upper_right = (2 * row + 1, 2 * column + 1)
            lower_left = (2 * row + 2, 2 * column)
            lower_right = (2 * row + 2, 2 * column + 1)
            edges.remove(_edge(upper_left, upper_right))
            edges.remove(_edge(lower_left, lower_right))
            edges.add(_edge(upper_left, lower_left))
            edges.add(_edge(upper_right, lower_right))

    inner_cells = [(row, column) for row in range(14) for column in range(14)]
    adjacency = {cell: [] for cell in inner_cells}
    for first, second in sorted(edges):
        adjacency[first].append(second)
        adjacency[second].append(first)
    if set(map(len, adjacency.values())) != {2}:
        raise AssertionError("macro splice did not form a cycle")

    anchor = (0, 13)
    removed_neighbor = rng.choice(sorted(adjacency[anchor]))
    edges.remove(_edge(anchor, removed_neighbor))
    adjacency = {cell: [] for cell in inner_cells}
    for first, second in sorted(edges):
        adjacency[first].append(second)
        adjacency[second].append(first)
    endpoints = [cell for cell, neighbors in adjacency.items() if len(neighbors) == 1]
    start = next(cell for cell in endpoints if cell != anchor)
    path = [start]
    previous = None
    current = start
    while current != anchor:
        onward = [cell for cell in adjacency[current] if cell != previous]
        if len(onward) != 1:
            raise AssertionError("opened macro-cycle is not a simple path")
        previous, current = current, onward[0]
        path.append(current)

    # Attach the 29 cells not covered by the 14x14 macro-cycle.
    path.extend((row, 14) for row in range(15))
    path.extend((14, column) for column in range(13, -1, -1))
    if len(path) != N * N or len(set(path)) != N * N:
        raise AssertionError("odd annex does not cover the 15x15 board exactly")
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(path, path[1:])):
        raise AssertionError("odd annex path contains a non-grid edge")

    rotation = rng.randrange(4)
    reflection = bool(rng.randrange(2))
    path = [board_core.transform_cell(cell, N, rotation, reflection) for cell in path]
    reversed_path = bool(rng.randrange(2))
    if reversed_path:
        path.reverse()
    tree_rows = [[list(first), list(second)] for first, second in tree]
    return path, {
        "program_family": "prim_macrocycle_odd_annex",
        "macro_grid_n": macro_n,
        "prim_root": list(root),
        "prim_tree_edges": tree_rows,
        "prim_tree_sha256": hashlib.sha256(
            json.dumps(tree_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "opened_anchor": list(anchor),
        "removed_neighbor": list(removed_neighbor),
        "annex_shape": "outer_row_plus_column_L",
        "d4_rotation_quarters": rotation,
        "d4_reflection": reflection,
        "path_reversed": reversed_path,
    }


def candidate_spec(case_index: int) -> board_core.Spec:
    if not 0 <= case_index < 16:
        raise ValueError("the balanced candidate pilot has case_index 0..15")
    form_index, rep = divmod(case_index, 4)
    form = board_core.QUESTION_FORMS[form_index]
    scope, statistic = board_core.FORM_MEANING[form]
    return board_core.Spec(
        index=case_index,
        block=case_index,
        question_form=form,
        scope=scope,
        statistic=statistic,
        layout_family="current_sparse_long",
        tier="xxhard",
        order=ORDER_BY_REP[rep] if scope == "all" else None,
        reference=REFERENCE_BY_REP[rep] if scope == "one" else None,
        dataset_role="ood_eval",
        nuisance_row=rep,
        n_variant=0,
    )


def _segments_from_lengths(path: Sequence[tuple[int, int]], lengths: Sequence[int]):
    segments = []
    cursor = 0
    for length in lengths:
        segments.append(list(path[cursor: cursor + length]))
        cursor += length
    if cursor != len(path):
        raise AssertionError("segment lengths do not consume the Hamiltonian path")
    return segments


def _distractors(segments: Sequence[Sequence[tuple[int, int]]], target: int):
    adjacency = board_core.pipe_adjacency(segments)
    return [
        index for index, segment in enumerate(segments)
        if index != target and index not in adjacency[target] and len(segment) >= 10
    ]


def build_semantic_pair(seed: int, case_index: int, max_attempts: int = 400) -> SemanticPair:
    """Build matched-length Backbite and alternative-program boards."""
    spec = candidate_spec(case_index)
    for attempt in range(max_attempts):
        control_rng = ood_rng_for(seed, case_index, "control", attempt)
        control_path = backbite.backbite_hamiltonian(N, control_rng, length_ladder.TORT)
        control_segments = [
            [tuple(cell) for cell in segment]
            for segment in backbite.cut_into_segments(control_path, PIPES, control_rng)
        ]
        lengths = [len(segment) for segment in control_segments]
        target_candidates = [
            index for index, length in enumerate(lengths)
            if TARGET_BAND[0] <= length <= TARGET_BAND[1]
            and _distractors(control_segments, index)
        ]
        if not target_candidates:
            continue
        target = control_rng.choice(target_candidates)

        for program_attempt in range(24):
            program_rng = ood_rng_for(
                seed, case_index, "program", attempt * 24 + program_attempt,
            )
            program_path, program_meta = prim_macrocycle_odd_annex(program_rng)
            program_segments = _segments_from_lengths(program_path, lengths)
            program_distractors = _distractors(program_segments, target)
            if not program_distractors:
                continue
            control_distractors = _distractors(control_segments, target)
            common = sorted(set(control_distractors).intersection(program_distractors))
            if not common:
                # A strict A/B ablation also holds the repeated-colour pipe
                # indices fixed.  Keep searching instead of accepting a board
                # whose palette placement exposes the program arm.
                continue
            control_distractor = program_distractor = control_rng.choice(common)

            board_core.validate_base(N, control_segments)
            board_core.validate_base(N, program_segments)
            control = board_core.BaseBoard(
                n=N,
                segments=control_segments,
                target_index=target,
                distractor_index=control_distractor,
                tortuosity=length_ladder.TORT,
                path_turn_fraction=round(backbite.turn_frac(control_path), 6),
                attempts=attempt + 1,
                base_board_hash=board_core.canonical_board_hash(N, control_segments),
                recipe_origin="xxhard_control",
            )
            program = board_core.BaseBoard(
                n=N,
                segments=program_segments,
                target_index=target,
                distractor_index=program_distractor,
                tortuosity=0.0,
                path_turn_fraction=round(backbite.turn_frac(program_path), 6),
                attempts=program_attempt + 1,
                base_board_hash=board_core.canonical_board_hash(N, program_segments),
                recipe_origin="ood_prim_macrocycle",
            )
            if control.base_board_hash == program.base_board_hash:
                continue
            return SemanticPair(
                spec=spec,
                control=control,
                program=program,
                control_program_meta={
                    "program_family": "backbite_hamiltonian",
                    "tortuosity": length_ladder.TORT,
                },
                program_meta=program_meta,
                attempts=attempt + 1,
            )
    raise RuntimeError(f"could not build OOD semantic pair for case {case_index}")


def _rounded_points(points: Sequence[tuple[float, float]], radius: float):
    if len(points) < 3:
        return list(points)
    out = [points[0]]
    for previous, current, following in zip(points, points[1:], points[2:]):
        in_dx, in_dy = current[0] - previous[0], current[1] - previous[1]
        out_dx, out_dy = following[0] - current[0], following[1] - current[1]
        in_len = math.hypot(in_dx, in_dy)
        out_len = math.hypot(out_dx, out_dy)
        if in_dx * out_dy == in_dy * out_dx:
            out.append(current)
            continue
        before = current[0] - radius * in_dx / in_len, current[1] - radius * in_dy / in_len
        after = current[0] + radius * out_dx / out_len, current[1] + radius * out_dy / out_len
        out.append(before)
        for step in range(1, 7):
            t = step / 7
            omt = 1 - t
            out.append((
                omt * omt * before[0] + 2 * omt * t * current[0] + t * t * after[0],
                omt * omt * before[1] + 2 * omt * t * current[1] + t * t * after[1],
            ))
        out.append(after)
    out.append(points[-1])
    return out


def _sine_points(points: Sequence[tuple[float, float]], amplitude: float):
    out = []
    for edge_index, (first, second) in enumerate(zip(points, points[1:])):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        for step in range(13):
            if edge_index and step == 0:
                continue
            t = step / 12
            # Two full waves: displacement is zero at both cell centres and at
            # their shared-boundary midpoint (t=.5).
            offset = amplitude * math.sin(4 * math.pi * t)
            out.append((first[0] + dx * t + nx * offset, first[1] + dy * t + ny * offset))
    return out


def _procedural_pictogram(name: str, side: int, opacity: int) -> Image.Image:
    """Return an alpha mask for an OOD-only pictorial silhouette.

    Drawing and rotating straight-alpha RGBA with bicubic interpolation can
    overshoot both RGB and alpha.  Keeping a scalar mask, rotating it with
    bilinear interpolation, and applying one solid tint afterwards makes the
    ``alpha <= opacity`` assumption exact.
    """
    tile = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(tile)
    fill = opacity
    s = side
    if name == "leaf":
        draw.ellipse([.18*s, .08*s, .82*s, .92*s], fill=fill)
        draw.line([(.28*s, .80*s), (.72*s, .20*s)], fill=fill, width=max(1, s//12))
    elif name == "fish":
        draw.ellipse([.15*s, .25*s, .75*s, .75*s], fill=fill)
        draw.polygon([(.70*s, .50*s), (.96*s, .22*s), (.96*s, .78*s)], fill=fill)
    elif name == "cloud":
        draw.ellipse([.08*s, .35*s, .50*s, .78*s], fill=fill)
        draw.ellipse([.30*s, .16*s, .72*s, .76*s], fill=fill)
        draw.ellipse([.55*s, .35*s, .94*s, .78*s], fill=fill)
        draw.rectangle([.18*s, .52*s, .84*s, .78*s], fill=fill)
    elif name == "butterfly":
        draw.ellipse([.05*s, .10*s, .48*s, .52*s], fill=fill)
        draw.ellipse([.52*s, .10*s, .95*s, .52*s], fill=fill)
        draw.ellipse([.12*s, .48*s, .48*s, .90*s], fill=fill)
        draw.ellipse([.52*s, .48*s, .88*s, .90*s], fill=fill)
        draw.ellipse([.45*s, .15*s, .55*s, .88*s], fill=fill)
    elif name == "apple":
        draw.ellipse([.14*s, .22*s, .86*s, .90*s], fill=fill)
        draw.ellipse([.50*s, .02*s, .82*s, .30*s], fill=fill)
    elif name == "umbrella":
        draw.pieslice([.06*s, .04*s, .94*s, .82*s], 180, 360, fill=fill)
        draw.line([(.50*s, .42*s), (.50*s, .88*s), (.68*s, .94*s)],
                  fill=fill, width=max(1, s//12), joint="curve")
    else:  # pragma: no cover - fixed table above
        raise ValueError(name)
    return tile


def _draw_pictogram_underlay(
    canvas: Image.Image,
    rng: random.Random,
    scale: int,
    size: int,
    count: int,
    tint: tuple[int, int, int],
) -> list[dict[str, Any]]:
    board_margin = int(size * 0.05)
    boxes: list[tuple[int, int, int, int]] = []
    rows = []
    used = set()
    for _ in range(count):
        for _attempt in range(100):
            name = PICTOGRAMS[rng.randrange(len(PICTOGRAMS))]
            if name in used:
                continue
            # Visible enough to constitute a real appearance shift, but kept
            # translucent and below the grid/pipes so it cannot delete path
            # evidence or endpoint annotations.
            tile_px = rng.randint(40, 54)
            left = rng.randint(board_margin, size - board_margin - tile_px)
            top = rng.randint(board_margin, size - board_margin - tile_px)
            box = (left, top, left + tile_px, top + tile_px)
            if any(not (box[2] + 4 < old[0] or old[2] + 4 < box[0]
                       or box[3] + 4 < old[1] or old[3] + 4 < box[1]) for old in boxes):
                continue
            opacity = rng.randint(60, 76)
            mask = _procedural_pictogram(name, tile_px * scale, opacity)
            angle = rng.uniform(-15.0, 15.0)
            mask = mask.rotate(angle, Image.Resampling.BILINEAR, expand=False)
            # Defensive clamp documents and enforces the renderer contract
            # even if Pillow's resampler changes in a future release.
            mask = mask.point(lambda value: min(value, opacity))
            tile = Image.new("RGBA", mask.size, (*tint, 0))
            tile.putalpha(mask)
            canvas.alpha_composite(tile, (left * scale, top * scale))
            boxes.append(box)
            used.add(name)
            rows.append({
                "id": f"procedural_ood/{name}",
                "name": name,
                "normalized_identity": name,
                "split": "ood_only",
                "style": "pictorial_silhouette",
                "bbox": list(box),
                "opacity_255": opacity,
                "rotation_deg": round(angle, 2),
            })
            break
        else:
            raise RuntimeError("could not place non-overlapping OOD icons")
    return rows


def render_ood_board(
    base: board_core.BaseBoard,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    size: int,
    seed: int,
    appearance_pair_index: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render a safe appearance shift; all clutter remains below the pipes."""
    scale = 3
    rng = ood_rng_for(seed, appearance_pair_index, "appearance_render")
    style = OOD_PIPE_STYLE
    surface_index = OOD_SURFACE_BY_PAIR[appearance_pair_index % len(OOD_SURFACE_BY_PAIR)]
    surface_name, surface_colors, grid_palette, pictogram_tint = OOD_SURFACE_FAMILIES[
        surface_index
    ]
    background = surface_colors[rng.randrange(len(surface_colors))]
    grid_rgb = grid_palette[rng.randrange(len(grid_palette))]
    canvas = Image.new("RGBA", (size * scale, size * scale), (*background, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    patches = []
    for _ in range(rng.randint(6, 9)):
        color = surface_colors[rng.randrange(len(surface_colors))]
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        points = []
        vertices = rng.randint(5, 8)
        for vertex in range(vertices):
            angle = 2 * math.pi * vertex / vertices + rng.uniform(-0.25, 0.25)
            radius = rng.uniform(70, 175)
            points.append(((cx + radius * math.cos(angle)) * scale,
                           (cy + radius * math.sin(angle)) * scale))
        draw.polygon(points, fill=(*color, 255))
        patches.append({"color_rgb": list(color), "vertices": [[round(x / scale, 1), round(y / scale, 1)] for x, y in points]})

    pictogram_count = 6
    pictograms = _draw_pictogram_underlay(
        canvas, rng, scale, size, pictogram_count, pictogram_tint
    )

    margin = int(size * 0.05)
    board = size - 2 * margin
    cell = board / base.n

    def center(cell_rc: Sequence[int]):
        row, column = int(cell_rc[0]), int(cell_rc[1])
        return ((margin + (column + 0.5) * cell) * scale,
                (margin + (row + 0.5) * cell) * scale)

    for index in range(base.n + 1):
        value = (margin + index * cell) * scale
        draw.line([(value, margin * scale), (value, (margin + board) * scale)],
                  fill=(*grid_rgb, 255), width=scale)
        draw.line([(margin * scale, value), ((margin + board) * scale, value)],
                  fill=(*grid_rgb, 255), width=scale)

    pipe_width = max(8 * scale, int(cell * 0.34 * scale))
    cap = int(cell * 0.32 * scale)
    per_pipe_styles = [
        OOD_COMPONENT_STYLES[index % len(OOD_COMPONENT_STYLES)]
        for index in range(len(base.segments))
    ]
    # Most non-colour-name questions deliberately contain one non-adjacent
    # same-colour distractor. Giving those two pipes different curve grammars
    # would reveal which route to follow without tracing it. On the fixed
    # four-pipe OOD board, assign the repeated pair one grammar and the two
    # remaining pipes the other; globally unique palettes retain a shuffled
    # 2/2 assignment.
    groups: dict[tuple[str, tuple[int, int, int]], list[int]] = {}
    for pipe_index, (name, rgb) in enumerate(colors):
        groups.setdefault((name, tuple(rgb)), []).append(pipe_index)
    repeated_groups = [indices for indices in groups.values() if len(indices) > 1]
    if repeated_groups:
        if (
            len(base.segments) != 4
            or len(repeated_groups) != 1
            or len(repeated_groups[0]) != 2
        ):
            raise ValueError("unsupported repeated-colour structure for OOD curves")
        repeated_style = OOD_COMPONENT_STYLES[rng.randrange(2)]
        other_style = next(
            style for style in OOD_COMPONENT_STYLES if style != repeated_style
        )
        repeated_indices = set(repeated_groups[0])
        per_pipe_styles = [
            repeated_style if index in repeated_indices else other_style
            for index in range(len(base.segments))
        ]
    else:
        rng.shuffle(per_pipe_styles)
    for segment, (_, rgb), pipe_style in zip(base.segments, colors, per_pipe_styles):
        centers = [center(cell_rc) for cell_rc in segment]
        if pipe_style == "rounded_corner":
            points = _rounded_points(centers, cell * 0.23 * scale)
        else:
            points = _sine_points(centers, cell * 0.08 * scale)
        draw.line(points, fill=(*rgb, 255), width=pipe_width, joint="curve")
        for x, y in (centers[0], centers[-1]):
            draw.rectangle([x - cap, y - cap, x + cap, y + cap], fill=(*rgb, 255))
            draw.rectangle(
                [x - cap + 3 * scale, y - cap + 3 * scale,
                 x + cap - 3 * scale, y + cap - 3 * scale],
                outline=tuple(int(value * 0.6) for value in rgb) + (255,),
                width=2 * scale,
            )

    image = canvas.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    # Pictogram antialiasing can produce every alpha between transparent and
    # the maximum requested opacity.  Include those actual underlay mixtures
    # in both safety gates instead of checking only the palette anchors.
    underlay_colors = [
        tuple(
            round((255 - alpha) * channel / 255 + alpha * tint_channel / 255)
            for channel, tint_channel in zip(bg, pictogram_tint)
        )
        for bg in surface_colors
        for alpha in range(77)
    ]
    min_contrast = min(
        math.dist(pipe_colors.lab(bg), pipe_colors.lab(rgb))
        for bg in underlay_colors for _, rgb in colors
    )
    min_grid_contrast = min(
        math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(rgb))
        for _, rgb in colors
    )
    min_grid_surface_contrast = min(
        math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(bg))
        for bg in underlay_colors
    )
    if min_contrast < 40:
        raise ValueError(f"OOD background contrast gate failed: {min_contrast:.2f}")
    if min_grid_contrast < 30:
        raise ValueError(f"OOD grid contrast gate failed: {min_grid_contrast:.2f}")
    if min_grid_surface_contrast < 18:
        raise ValueError(
            "OOD grid/surface contrast gate failed: "
            f"{min_grid_surface_contrast:.2f}"
        )

    # Every true cell-to-cell connection crosses its shared boundary at the
    # midpoint.  The rounded and sine programs are explicitly port-locked.
    pixels = image.load()
    for segment, (_, rgb) in zip(base.segments, colors):
        for first, second in zip(segment, segment[1:]):
            a = center(first)
            b = center(second)
            x = int(round((a[0] + b[0]) / (2 * scale)))
            y = int(round((a[1] + b[1]) / (2 * scale)))
            got = pixels[x, y]
            if max(abs(got[channel] - rgb[channel]) for channel in range(3)) > 36:
                raise ValueError("OOD renderer missed a true grid-boundary port")
    return image, {
        "appearance_family": f"ood_confounded/{style}/{surface_name}_pictograms",
        "surface_family": surface_name,
        "pipe_style": style,
        "per_pipe_styles": per_pipe_styles,
        "same_color_curve_style_tied": all(
            len({per_pipe_styles[index] for index in indices}) == 1
            for indices in groups.values()
        ),
        "background_rgb": list(background),
        "grid_rgb": list(grid_rgb),
        "pictogram_tint_rgb": list(pictogram_tint),
        "texture_style": "non_grid_aligned_patchwork",
        "patches": patches,
        "distractor_objects": pictograms,
        "distractor_object_count": len(pictograms),
        "distractor_object_partition": "procedural_ood_only",
        "distractor_objects_are_underlay": True,
        "min_pipe_background_de": round(min_contrast, 2),
        "min_pipe_grid_de": round(min_grid_contrast, 2),
        "min_grid_surface_de": round(min_grid_surface_contrast, 2),
        "pre_annotation_port_midpoint_recall": 1.0,
        "supersample": scale,
        "compositing_order": [
            "patchwork", "procedural_ood_pictograms", "grid", "pipes",
            "endpoint_annotations",
        ],
    }


def _segment_rows(base, colors, labeling, label_annotations):
    rows = []
    for pipe_index, (segment, color) in enumerate(zip(base.segments, colors)):
        label_row = next((row for row in label_annotations if row["pipe_index"] == pipe_index), None)
        rows.append({
            "pipe_id": f"pipe{pipe_index}",
            "pipe_index": pipe_index,
            "cells": [[row, column] for row, column in segment],
            "length": len(segment),
            "turn_count": board_core.turns(segment),
            "color_name": color[0],
            "color_rgb": list(color[1]),
            "display_label": labeling.segment_to_label[pipe_index] if labeling else None,
            "label_cell": label_row["cell"] if label_row else None,
            "label_endpoint_side": label_row["endpoint_side"] if label_row else None,
            "label_bbox": label_row["bbox"] if label_row else None,
        })
    return rows


def build_record(
    seed: int,
    case_index: int,
    arm: str,
    pair: SemanticPair,
    size: int = SIZE,
) -> tuple[dict[str, Any], Image.Image]:
    if arm not in ARMS:
        raise ValueError(f"unknown OOD arm: {arm}")
    spec = pair.spec
    rep = case_index % 4
    base = pair.program if arm in PROGRAM_ARMS else pair.control
    semantic_seed = f"{RNG_NAMESPACE}|ood-shared|{seed}"
    # Pair every visible nuisance across length and turn-count cases inside a
    # scope.  Geometry remains independent; colour multiset, label RNG,
    # background, patchwork, pictograms, and line style do not reveal which
    # statistic the question asks for.
    nuisance_pair_index = rep + (4 if spec.scope == "one" else 0)
    colors, palette_mode, min_color_de = board_core.assign_colors(
        semantic_seed, nuisance_pair_index, spec, base
    )
    labeling = (
        board_core.assign_labels(semantic_seed, nuisance_pair_index, base)
        if spec.scope == "all" else None
    )
    appearance_spec = dataclasses.replace(
        spec, block=rep, dataset_role="train_id"
    )
    id_appearance = board_render.appearance_for(semantic_seed, appearance_spec)
    if arm in APPEARANCE_ARMS:
        image, appearance_meta = render_ood_board(
            base, colors, size, seed, nuisance_pair_index
        )
        label_annotations = board_core.draw_labels(image, base, labeling, size) if labeling else []
        ring_annotation = None
        if spec.scope == "one" and spec.reference == "circle":
            ring_annotation = board_core.draw_ring(image, base.segments[base.target_index][0], base.n, size)
    else:
        image, label_annotations, ring_annotation = board_core.render_problem(
            semantic_seed, case_index, spec, base, colors, labeling, id_appearance, size
        )
        appearance_meta = board_render.metadata(id_appearance, colors, pipe_colors.lab)

    appearance_meta["final_port_midpoint_recall"] = round(
        board_core.port_midpoint_recall(
            image, base.n, base.segments, [rgb for _, rgb in colors], size
        ),
        6,
    )

    answer = board_core.answer_for(base, spec, labeling)
    question, template_id = board_core.question_for(semantic_seed, case_index, spec, base, colors)
    if spec.statistic == "turn_count":
        question = question.replace(
            "endpoint squares never count as turns.",
            "endpoint squares never count as turns. Decorative curvature "
            "within a square is not an additional turn.",
        )
        template_id += "_ood_curve_rule"
    lengths = [len(segment) for segment in base.segments]
    turn_counts = [board_core.turns(segment) for segment in base.segments]
    rows = _segment_rows(base, colors, labeling, label_annotations)
    program_meta = pair.program_meta if arm in PROGRAM_ARMS else pair.control_program_meta
    image_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    target_color = colors[base.target_index][0]
    layout_family = (
        "ood_prim_macrocycle" if arm in PROGRAM_ARMS else "current_sparse_long"
    )
    record_index = case_index * len(ARMS) + ARMS.index(arm)
    meta = {
        "schema_version": RNG_NAMESPACE,
        "base_generator_version": board_core.OOD_COMPAT_VERSION,
        "rng_namespace_version": RNG_NAMESPACE,
        "generation_seed": seed,
        "global_index": record_index,
        "case_index": case_index,
        "dataset_role": "ood_eval_candidate",
        "split": "eval",
        "is_ood": True,
        "difficulty": "xxhard",
        "tier": "xxhard",
        "ood_arm": arm,
        "paired_case_id": f"case_{case_index:02d}",
        "nuisance_pair_key": f"{spec.scope}:{rep}",
        "nuisance_pair_index": nuisance_pair_index,
        "semantic_pair_member": "program" if arm in PROGRAM_ARMS else "control",
        "render_pair_member": "ood" if arm in APPEARANCE_ARMS else "id",
        "difficulty_calibration": "provisional_ood_candidate_not_calibrated_harder",
        "scope": spec.scope,
        "statistic": spec.statistic,
        "order": spec.order,
        "reference": spec.reference,
        "question_form": spec.question_form,
        "layout_family": layout_family,
        "grid_n": base.n,
        "num_pipes": len(base.segments),
        "num_colors": len({color[0] for color in colors}),
        "palette_mode": palette_mode,
        "min_pairwise_color_de": min_color_de,
        "pairwise_color_de_gate": board_core.color_de_gate(
            len({color[0] for color in colors})
        ),
        "path_program": program_meta["program_family"],
        "path_program_meta": program_meta,
        "path_turn_fraction": base.path_turn_fraction,
        "base_board_hash": base.base_board_hash,
        "image_pixel_sha256": image_hash,
        "image_size": size,
        "all_lengths": lengths,
        "all_turns": turn_counts,
        "target_index": base.target_index,
        "target_pipe_id": f"pipe{base.target_index}",
        "target_color": target_color,
        "distractor_index": base.distractor_index,
        "distractor_pipe_id": f"pipe{base.distractor_index}",
        "distractor_color": colors[base.distractor_index][0],
        "segments": rows,
        "labels_present": labeling is not None,
        "label_to_segment": ({str(key): value for key, value in sorted(labeling.label_to_segment.items())}
                             if labeling else {}),
        "label_annotations": label_annotations,
        "ring_present": ring_annotation is not None,
        "ring": ring_annotation,
        "template_id": template_id,
        "pair_build_attempts": pair.attempts,
        **appearance_meta,
    }
    record_id = f"flow_free_ood_candidate_{case_index:02d}_{arm}"
    record = {
        "id": record_id,
        "source_id": record_id,
        "task": "flow_free",
        "task_type": "visual_tracking",
        "subtype": "lines_observation",
        "level": "fine",
        "difficulty": "xxhard",
        "tier": "xxhard",
        "question_form": spec.question_form,
        "scope": spec.scope,
        "statistic": spec.statistic,
        "layout_family": layout_family,
        "order": spec.order,
        "reference": spec.reference,
        "question": question,
        "ans_type": "blank",
        "options": [],
        "answer": answer,
        "cot": board_core.cot_for(base, spec, answer, labeling),
        "image": f"images/{case_index:02d}_{arm}.png",
        "meta": meta,
    }
    validate_candidate(record, image)
    return record, image


def validate_candidate(record: dict[str, Any], image: Image.Image | None = None) -> None:
    meta = record["meta"]
    if meta.get("dataset_role") != "ood_eval_candidate" or record.get("tier") != "xxhard":
        raise ValueError("OOD candidate role/tier mismatch")
    if meta.get("schema_version") != RNG_NAMESPACE or meta.get("rng_namespace_version") != RNG_NAMESPACE:
        raise ValueError("OOD candidate version metadata mismatch")
    if meta.get("base_generator_version") != board_core.OOD_COMPAT_VERSION:
        raise ValueError("OOD candidate base-generator version mismatch")
    for field in (
        "difficulty", "tier", "question_form", "scope", "statistic",
        "layout_family", "order", "reference",
    ):
        if record.get(field) != meta.get(field):
            raise ValueError(f"OOD top-level/meta field mismatch: {field}")
    case_index = int(meta.get("case_index", -1))
    arm = meta.get("ood_arm")
    if case_index not in range(16) or arm not in ARMS:
        raise ValueError("unknown OOD case or arm")
    expected_spec = candidate_spec(case_index)
    expected_layout = (
        "ood_prim_macrocycle" if arm in PROGRAM_ARMS else "current_sparse_long"
    )
    for field in ("question_form", "scope", "statistic", "order", "reference"):
        if record.get(field) != getattr(expected_spec, field):
            raise ValueError(f"OOD record field disagrees with candidate schedule: {field}")
    if record.get("layout_family") != expected_layout:
        raise ValueError("OOD layout family disagrees with arm")
    if record.get("difficulty") != "xxhard" or meta.get("difficulty") != "xxhard":
        raise ValueError("OOD candidate difficulty mismatch")
    if meta.get("split") != "eval" or meta.get("is_ood") is not True:
        raise ValueError("OOD candidate split/is_ood mismatch")
    expected_semantic_member = "program" if arm in PROGRAM_ARMS else "control"
    expected_render_member = "ood" if arm in APPEARANCE_ARMS else "id"
    if meta.get("semantic_pair_member") != expected_semantic_member:
        raise ValueError("OOD semantic pair member disagrees with arm")
    if meta.get("render_pair_member") != expected_render_member:
        raise ValueError("OOD render pair member disagrees with arm")
    rep = case_index % 4
    if meta.get("nuisance_pair_key") != f"{expected_spec.scope}:{rep}":
        raise ValueError("OOD nuisance pair key disagrees with schedule")
    expected_nuisance_index = rep + (4 if expected_spec.scope == "one" else 0)
    if int(meta.get("nuisance_pair_index", -1)) != expected_nuisance_index:
        raise ValueError("OOD nuisance pair index disagrees with schedule")
    if meta.get("paired_case_id") != f"case_{case_index:02d}":
        raise ValueError("OOD paired case id disagrees with schedule")
    expected_global = case_index * len(ARMS) + ARMS.index(arm)
    if int(meta.get("global_index", -1)) != expected_global:
        raise ValueError("OOD global index is not unique case x arm index")
    expected_id = f"flow_free_ood_candidate_{case_index:02d}_{arm}"
    if record.get("id") != expected_id or record.get("source_id") != expected_id:
        raise ValueError("OOD id namespace mismatch")
    if record.get("image") != f"images/{case_index:02d}_{arm}.png":
        raise ValueError("OOD image namespace mismatch")
    rows = meta["segments"]
    segments = [[tuple(cell) for cell in row["cells"]] for row in rows]
    board_core.validate_base(int(meta["grid_n"]), segments)
    if int(meta["grid_n"]) != N or int(meta["num_pipes"]) != PIPES:
        raise ValueError("OOD candidate board size/pipes mismatch")
    lengths = [len(segment) for segment in segments]
    turns = [board_core.turns(segment) for segment in segments]
    if lengths != meta["all_lengths"] or turns != meta["all_turns"]:
        raise ValueError("OOD candidate geometry statistics disagree")
    if len(rows) != int(meta["num_pipes"]):
        raise ValueError("OOD num_pipes disagrees with segment rows")
    for pipe_index, (row, segment, length, turn_count) in enumerate(
        zip(rows, segments, lengths, turns)
    ):
        if row.get("pipe_index") != pipe_index or row.get("pipe_id") != f"pipe{pipe_index}":
            raise ValueError("OOD pipe row identity mismatch")
        if row.get("length") != length or row.get("turn_count") != turn_count:
            raise ValueError("OOD pipe row statistic mismatch")
        rgb = row.get("color_rgb")
        if not (
            isinstance(rgb, list) and len(rgb) == 3
            and all(isinstance(value, int) and 0 <= value <= 255 for value in rgb)
        ):
            raise ValueError("OOD pipe row has an invalid RGB colour")
    full_path = [cell for segment in segments for cell in segment]
    if any(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
        for first, second in zip(full_path, full_path[1:])
    ):
        raise ValueError("OOD segment ordering no longer forms its source Hamiltonian path")
    if round(backbite.turn_frac(full_path), 6) != float(meta["path_turn_fraction"]):
        raise ValueError("OOD path turn fraction disagrees with geometry")
    if board_core.canonical_board_hash(N, segments) != meta["base_board_hash"]:
        raise ValueError("OOD candidate board hash disagrees")
    if meta["path_program"] not in {"backbite_hamiltonian", "prim_macrocycle_odd_annex"}:
        raise ValueError("unknown OOD candidate path program")
    if float(meta["min_pipe_background_de"]) < 40:
        raise ValueError("OOD candidate contrast gate failed")
    if float(meta.get("min_pipe_grid_de", 0.0)) < 30:
        raise ValueError("OOD candidate pipe/grid contrast gate failed")
    expected_color_gate = board_core.color_de_gate(int(meta.get("num_colors", 0)))
    if float(meta.get("pairwise_color_de_gate", -1.0)) != expected_color_gate:
        raise ValueError("OOD candidate pipe-colour dE gate metadata disagrees")
    if float(meta.get("min_pairwise_color_de", 0.0)) < expected_color_gate:
        raise ValueError("OOD candidate pipe colours are below the safety gate")
    expected_program = (
        "prim_macrocycle_odd_annex"
        if arm in PROGRAM_ARMS else "backbite_hamiltonian"
    )
    if meta["path_program"] != expected_program:
        raise ValueError("OOD arm and path program disagree")
    program_meta = meta.get("path_program_meta") or {}
    if program_meta.get("program_family") != meta["path_program"]:
        raise ValueError("OOD path-program provenance disagrees")
    if meta["path_program"] == "prim_macrocycle_odd_annex":
        tree_rows = program_meta.get("prim_tree_edges")
        if not isinstance(tree_rows, list) or len(tree_rows) != 48:
            raise ValueError("OOD Prim provenance has a malformed tree")
        tree_hash = hashlib.sha256(
            json.dumps(tree_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if tree_hash != program_meta.get("prim_tree_sha256"):
            raise ValueError("OOD Prim tree provenance hash disagrees")
    if arm in APPEARANCE_ARMS:
        surface_index = OOD_SURFACE_BY_PAIR[
            expected_nuisance_index % len(OOD_SURFACE_BY_PAIR)
        ]
        expected_surface, surface_colors, grid_palette, expected_tint = (
            OOD_SURFACE_FAMILIES[surface_index]
        )
        if meta.get("surface_family") != expected_surface:
            raise ValueError("appearance-shift arm has the wrong surface family")
        if tuple(meta.get("background_rgb", ())) not in surface_colors:
            raise ValueError("appearance-shift background is outside its surface palette")
        if tuple(meta.get("grid_rgb", ())) not in grid_palette:
            raise ValueError("appearance-shift grid ink disagrees with its surface family")
        if tuple(meta.get("pictogram_tint_rgb", ())) != expected_tint:
            raise ValueError("appearance-shift pictogram tint disagrees with its surface family")
        grid_rgb = tuple(meta["grid_rgb"])
        pipe_rgbs = [tuple(row["color_rgb"]) for row in rows]
        underlay_colors = [
            tuple(
                round(
                    (255 - alpha) * channel / 255
                    + alpha * tint_channel / 255
                )
                for channel, tint_channel in zip(bg, expected_tint)
            )
            for bg in surface_colors
            for alpha in range(77)
        ]
        measured_pipe_surface = round(min(
            math.dist(pipe_colors.lab(bg), pipe_colors.lab(rgb))
            for bg in underlay_colors for rgb in pipe_rgbs
        ), 2)
        measured_pipe_grid = round(min(
            math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(rgb))
            for rgb in pipe_rgbs
        ), 2)
        measured_grid_surface = round(min(
            math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(bg))
            for bg in underlay_colors
        ), 2)
        for field, measured in (
            ("min_pipe_background_de", measured_pipe_surface),
            ("min_pipe_grid_de", measured_pipe_grid),
            ("min_grid_surface_de", measured_grid_surface),
        ):
            if float(meta.get(field, -1.0)) != measured:
                raise ValueError(f"OOD stored appearance contrast disagrees: {field}")
        if measured_pipe_surface < 40 or measured_pipe_grid < 30:
            raise ValueError("appearance-shift pipe contrast is unsafe")
        if measured_grid_surface < 18:
            raise ValueError("appearance-shift grid is not visible on its underlay")
        if meta.get("pipe_style") != OOD_PIPE_STYLE:
            raise ValueError("appearance-shift arm lacks an OOD line style")
        if sorted(meta.get("per_pipe_styles", [])) != sorted(
            OOD_COMPONENT_STYLES * 2
        ):
            raise ValueError("appearance-shift arm lacks a balanced style mix")
        if meta.get("same_color_curve_style_tied") is not True:
            raise ValueError("appearance-shift curve grammar leaks same-colour identity")
        style_by_color: dict[tuple[str, tuple[int, ...]], set[str]] = {}
        for row, component_style in zip(rows, meta["per_pipe_styles"]):
            key = (row["color_name"], tuple(row["color_rgb"]))
            style_by_color.setdefault(key, set()).add(component_style)
        if any(len(styles) != 1 for styles in style_by_color.values()):
            raise ValueError("same-colour pipes received different curve grammars")
        if not meta.get("distractor_objects_are_underlay"):
            raise ValueError("OOD distractor objects must remain underlays")
        if int(meta.get("distractor_object_count", 0)) < 5:
            raise ValueError("appearance-shift arm lacks distractor objects")
        if float(meta.get("pre_annotation_port_midpoint_recall", 0.0)) != 1.0:
            raise ValueError("appearance-shift renderer lost a true pipe port")
    else:
        expected_id_appearance = board_render.appearance_for(
            f"{RNG_NAMESPACE}|ood-shared|{meta.get('generation_seed')}",
            dataclasses.replace(
                expected_spec, block=rep, dataset_role="train_id"
            ),
        )
        for field, expected in (
            ("appearance_family", expected_id_appearance.family),
            ("background_rgb", list(expected_id_appearance.background_rgb)),
            ("grid_rgb", list(expected_id_appearance.grid_rgb)),
            ("texture_style", expected_id_appearance.texture_style),
            ("texture_amplitude_rgb", expected_id_appearance.texture_amplitude),
            ("texture_scale_px", expected_id_appearance.texture_scale_px),
            ("texture_seed", expected_id_appearance.texture_seed),
            ("rgb_jitter_bound", board_render.RGB_JITTER),
        ):
            if meta.get(field) != expected:
                raise ValueError(
                    f"OOD control appearance metadata disagrees: {field}"
                )
        if meta.get("pipe_style") != "orthogonal_solid":
            raise ValueError("control arm must use the ID pipe style")
        background_rgb = tuple(meta["background_rgb"])
        grid_rgb = tuple(meta["grid_rgb"])
        pipe_rgbs = [tuple(row["color_rgb"]) for row in rows]
        measured_pipe_surface = round(min(
            math.dist(pipe_colors.lab(background_rgb), pipe_colors.lab(rgb))
            for rgb in pipe_rgbs
        ), 2)
        measured_pipe_grid = round(min(
            math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(rgb))
            for rgb in pipe_rgbs
        ), 2)
        measured_grid_surface = round(
            math.dist(pipe_colors.lab(grid_rgb), pipe_colors.lab(background_rgb)), 2
        )
        for field, measured in (
            ("min_pipe_background_de", measured_pipe_surface),
            ("min_pipe_grid_de", measured_pipe_grid),
            ("grid_background_de", measured_grid_surface),
        ):
            if float(meta.get(field, -1.0)) != measured:
                raise ValueError(f"OOD control contrast disagrees: {field}")
        if measured_pipe_surface < 40 or measured_pipe_grid < 30:
            raise ValueError("OOD control pipe contrast is unsafe")
        if measured_grid_surface < 18:
            raise ValueError("OOD control grid is not visible")
    if float(meta.get("final_port_midpoint_recall", 0.0)) != 1.0:
        raise ValueError("final OOD image lost a true pipe port")

    spec = board_core.Spec(
        index=int(meta["case_index"]), block=int(meta["case_index"]),
        question_form=record["question_form"], scope=record["scope"],
        statistic=record["statistic"], layout_family=record["layout_family"],
        tier="xxhard", order=record["order"], reference=record["reference"],
        dataset_role="ood_eval",
    )
    base = board_core.BaseBoard(
        n=N, segments=segments, target_index=int(meta["target_index"]),
        distractor_index=int(meta["distractor_index"]), tortuosity=0.0,
        path_turn_fraction=float(meta["path_turn_fraction"]), attempts=1,
        base_board_hash=meta["base_board_hash"], recipe_origin="validator",
    )
    labeling = None
    if record["scope"] == "all":
        label_map = {int(key): int(value) for key, value in meta["label_to_segment"].items()}
        segment_to_label = [0] * len(segments)
        for label, segment_index in label_map.items():
            segment_to_label[segment_index] = label
        sides = [0 if row["label_endpoint_side"] == "start" else 1 for row in rows]
        labeling = board_core.Labeling(segment_to_label, sides)
    if board_core.answer_for(base, spec, labeling) != record["answer"]:
        raise ValueError("OOD candidate answer does not recompute")

    colors = [row["color_name"] for row in rows]
    target, distractor = base.target_index, base.distractor_index
    if len(set(colors)) != int(meta["num_colors"]):
        raise ValueError("OOD num_colors disagrees with pipe rows")
    distinct_rgbs = []
    known_colors = dict(pipe_colors.PIPE_COLORS)
    for row in rows:
        if known_colors.get(row["color_name"]) != tuple(row["color_rgb"]):
            raise ValueError("OOD colour name/RGB disagrees")
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
        raise ValueError("OOD min_pairwise_color_de disagrees with pipe rows")
    if meta.get("target_pipe_id") != f"pipe{target}" or meta.get("target_color") != colors[target]:
        raise ValueError("OOD target metadata disagrees")
    if (
        meta.get("distractor_pipe_id") != f"pipe{distractor}"
        or meta.get("distractor_color") != colors[distractor]
    ):
        raise ValueError("OOD distractor metadata disagrees")
    if record["scope"] == "one" and record["reference"] == "color_name":
        if len(colors) != len(set(colors)):
            raise ValueError("OOD color-name record has repeated colors")
    else:
        if colors[target] != colors[distractor]:
            raise ValueError("OOD repeated-color target/distractor disagree")
        adjacency = board_core.pipe_adjacency(segments)
        if distractor in adjacency[target]:
            raise ValueError("OOD repeated-color pipes touch")

    if record["scope"] == "all":
        label_map = {
            int(key): int(value) for key, value in meta["label_to_segment"].items()
        }
        if set(label_map) != set(range(1, len(segments) + 1)):
            raise ValueError("OOD labels are not a 1..num_pipes bijection")
        if set(label_map.values()) != set(range(len(segments))):
            raise ValueError("OOD labels do not cover every pipe")
        boxes = [row["label_bbox"] for row in rows]
        if any(box is None for box in boxes):
            raise ValueError("OOD all-pipe record is missing a label bbox")
        canvas_size = int(meta["image_size"])
        for index, (row, segment, box) in enumerate(zip(rows, segments, boxes)):
            if not (
                0 <= box[0] < box[2] < canvas_size
                and 0 <= box[1] < box[3] < canvas_size
            ):
                raise ValueError("OOD label lies outside the canvas")
            if tuple(row["label_cell"]) not in (segment[0], segment[-1]):
                raise ValueError("OOD label is not on its pipe endpoint")
            expected_label = next(
                label for label, segment_index in label_map.items()
                if segment_index == index
            )
            if row.get("display_label") != expected_label:
                raise ValueError("OOD row display label disagrees with label map")
            annotation = next(
                (item for item in meta["label_annotations"]
                 if item.get("pipe_index") == index),
                None,
            )
            if annotation is None or any(
                annotation.get(key) != row.get(row_key)
                for key, row_key in (
                    ("label", "display_label"), ("cell", "label_cell"),
                    ("endpoint_side", "label_endpoint_side"), ("bbox", "label_bbox"),
                )
            ):
                raise ValueError("OOD label annotation disagrees with pipe row")
            for other in boxes[index + 1:]:
                if board_core.rectangles_overlap(box, other):
                    raise ValueError("OOD number-label badges overlap")
    elif record["reference"] == "circle":
        if not meta.get("ring_present"):
            raise ValueError("OOD circle-reference record lacks a ring")
        if tuple(meta["ring"]["ring_cell"]) != segments[target][0]:
            raise ValueError("OOD ring is not on the target start endpoint")

    question = str(record.get("question", ""))
    if record["statistic"] == "length":
        if "endpoint" not in question.lower() or "square" not in question.lower():
            raise ValueError("OOD length question lacks endpoint-square semantics")
    else:
        lowered = question.lower()
        if not all(word in lowered for word in ("turn", "perpendicular", "never count")):
            raise ValueError("OOD turn question lacks its grid-turn definition")
        if "decorative curvature" not in lowered:
            raise ValueError("OOD turn question lacks its curve disambiguation")
    if record["scope"] == "one" and record["reference"] == "color_name":
        if colors[target] not in question:
            raise ValueError("OOD named-colour question lacks its target colour")

    if image is not None:
        if image.size != (int(meta["image_size"]), int(meta["image_size"])):
            raise ValueError("OOD image size disagrees")
        got_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if got_hash != meta["image_pixel_sha256"]:
            raise ValueError("OOD image hash disagrees")
        measured_recall = round(
            board_core.port_midpoint_recall(
                image, N, segments, [row["color_rgb"] for row in rows], image.width
            ),
            6,
        )
        if measured_recall != float(meta["final_port_midpoint_recall"]):
            raise ValueError("OOD final port recall metadata disagrees")
        if record["scope"] == "one" and record["reference"] == "circle":
            cx, cy = board_core.cell_center(segments[target][0], N, image.width)
            got = image.getpixel((int(round(cx)), int(round(cy))))
            target_rgb = tuple(rows[target]["color_rgb"])
            if max(abs(a - b) for a, b in zip(got, target_rgb)) > 60:
                raise ValueError("OOD ring center does not lie on the target pipe")
