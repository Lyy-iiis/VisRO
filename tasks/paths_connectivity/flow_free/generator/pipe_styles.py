"""Pipe-style renderer for Flow Free material variants.

This module does not change the standard board renderer or its record schema.  It holds geometry, palette, labels, question, and answer fixed
while varying three rendering knobs independently:

* centre-line width as a fraction of one grid cell;
* colour-layer opacity;
* a non-destructive texture clipped inside a continuous solid pipe mask.

Styles may be sampled once per board or independently once per pipe.  Patterns
use one global canvas phase for every pipe.  They never restart per cell, and
they modulate colour rather than punching transparent holes, so they cannot act
as a per-cell ruler.  Per-pipe choices are sampled independently of geometry,
target identity, question form, and answer.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageDraw

from . import board_core, board_render, pipe_colors

VERSION = "flow_free_pipe_style"
TEXTURES = ("solid", "dots", "diagonal_stripes", "crosshatch", "center_rail")


@dataclasses.dataclass(frozen=True)
class PipeStyle:
    name: str
    group: str
    width_fraction: float
    opacity_percent: int
    texture: str = "solid"
    core_fraction: float = 0.0
    core_opacity_percent: int = 100
    outline_fraction: float = 0.0
    outline_opacity_percent: int = 100
    halo_fraction: float = 0.0
    halo_opacity_percent: int = 0
    cell_fill_opacity_percent: int = 0
    grid_on_top: bool = False
    canonical_renderer: bool = False

    def validate(self) -> None:
        if not self.name or not self.group:
            raise ValueError("style name and group must be non-empty")
        if not 0.18 <= self.width_fraction <= 0.88:
            raise ValueError("ordinary pipe width must stay within 0.18--0.88 cell")
        if not 50 <= self.opacity_percent <= 100:
            raise ValueError("pipe opacity must stay within 50--100 percent")
        if self.texture not in TEXTURES:
            raise ValueError(f"unknown pipe texture: {self.texture}")
        if not 0.0 <= self.core_fraction <= 0.25:
            raise ValueError("core fraction is outside the supported range")
        if self.core_fraction >= self.width_fraction:
            raise ValueError("opaque core must be narrower than its pipe")
        if not 0.0 <= self.outline_fraction <= 0.12:
            raise ValueError("outline fraction is outside the supported range")
        if self.width_fraction + 2 * self.outline_fraction > 0.90:
            raise ValueError("pipe plus outline is too wide to preserve false boundaries")
        if not 0.0 <= self.halo_fraction <= 0.30:
            raise ValueError("halo fraction is outside the supported range")
        for value in (
            self.outline_opacity_percent,
            self.core_opacity_percent,
            self.halo_opacity_percent,
            self.cell_fill_opacity_percent,
        ):
            if not 0 <= value <= 100:
                raise ValueError("style opacity percentage is outside 0--100")


def _geometry(n: int, size: int) -> tuple[float, float, float]:
    margin = float(int(size * 0.05))
    board = float(size - 2 * int(size * 0.05))
    return margin, board, board / n


def _center(
    cell_rc: Sequence[int], n: int, size: int
) -> tuple[float, float]:
    margin, _, cell = _geometry(n, size)
    row, column = int(cell_rc[0]), int(cell_rc[1])
    return margin + (column + 0.5) * cell, margin + (row + 0.5) * cell


def _draw_grid(image: Image.Image, n: int, size: int, rgb: Sequence[int]) -> None:
    margin, board, cell = _geometry(n, size)
    draw = ImageDraw.Draw(image)
    for index in range(n + 1):
        value = margin + index * cell
        draw.line(
            [(value, margin), (value, margin + board)],
            fill=tuple(map(int, rgb)), width=1,
        )
        draw.line(
            [(margin, value), (margin + board, value)],
            fill=tuple(map(int, rgb)), width=1,
        )


def _path_mask(
    n: int,
    segment: Sequence[Sequence[int]],
    size: int,
    width_fraction: float,
    *,
    endpoint_caps: bool = True,
) -> tuple[Image.Image, int]:
    """Rasterize a continuous route mask without leaving its false ports."""
    margin, _, cell = _geometry(n, size)
    width = max(3, int(round(cell * width_fraction)))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    points = [_center(cell_rc, n, size) for cell_rc in segment]
    if len(points) > 1:
        draw.line(points, fill=255, width=width, joint="curve")
    radius = width / 2
    for x, y in points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    if endpoint_caps:
        cap = max(cell * 0.30, radius)
        for cell_rc, (x, y) in ((segment[0], points[0]), (segment[-1], points[-1])):
            row, column = map(int, cell_rc)
            left = margin + column * cell
            top = margin + row * cell
            right = left + cell
            bottom = top + cell
            box = [
                max(math.ceil(x - cap), math.ceil(left) + 1),
                max(math.ceil(y - cap), math.ceil(top) + 1),
                min(math.floor(x + cap), math.floor(right) - 1),
                min(math.floor(y + cap), math.floor(bottom) - 1),
            ]
            draw.rectangle(box, fill=255)
    return mask, width


def _opacity_mask(mask: Image.Image, opacity_percent: int) -> Image.Image:
    if opacity_percent == 100:
        return mask
    scale = opacity_percent / 100
    return mask.point(lambda value: round(value * scale))


def _paste(
    image: Image.Image,
    rgb: Sequence[int],
    mask: Image.Image,
    opacity_percent: int,
) -> None:
    image.paste(tuple(map(int, rgb)), (0, 0), _opacity_mask(mask, opacity_percent))


def _mix(first: Sequence[int], second: Sequence[int], weight: float) -> tuple[int, int, int]:
    return tuple(
        round((1.0 - weight) * int(a) + weight * int(b))
        for a, b in zip(first, second)
    )


def _accent(rgb: Sequence[int]) -> tuple[int, int, int]:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    anchor = (245, 245, 245) if luminance < 135 else (18, 20, 24)
    return _mix(rgb, anchor, 0.42)


def _adaptive_outline(background: Sequence[int]) -> tuple[int, int, int]:
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return (238, 240, 244) if luminance < 128 else (24, 27, 31)


def _pattern_mask(size: int, cell: float, texture: str) -> Image.Image:
    pattern = Image.new("L", (size, size), 0)
    if texture == "solid" or texture == "center_rail":
        return pattern
    draw = ImageDraw.Draw(pattern)
    spacing = max(8, int(round(cell * 0.62)))
    if texture == "dots":
        radius = max(1, int(round(spacing * 0.14)))
        phase = spacing // 3
        for y in range(phase, size + spacing, spacing):
            for x in range(phase, size + spacing, spacing):
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
        return pattern
    stripe_width = max(2, int(round(cell * 0.10)))
    for offset in range(-size, 2 * size, spacing):
        draw.line([(offset, 0), (offset - size, size)], fill=255, width=stripe_width)
    if texture == "crosshatch":
        for offset in range(-size, 2 * size, spacing):
            draw.line([(offset, 0), (offset + size, size)], fill=255, width=stripe_width)
    return pattern


def _cell_fill_mask(
    n: int, segment: Sequence[Sequence[int]], size: int
) -> Image.Image:
    margin, _, cell = _geometry(n, size)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for row, column in segment:
        left = math.ceil(margin + int(column) * cell) + 1
        top = math.ceil(margin + int(row) * cell) + 1
        right = math.floor(margin + (int(column) + 1) * cell) - 1
        bottom = math.floor(margin + (int(row) + 1) * cell) - 1
        draw.rectangle([left, top, right, bottom], fill=255)
    return mask


def _mask_topology(
    n: int,
    segments: Sequence[Sequence[Sequence[int]]],
    masks: Sequence[Image.Image],
    size: int,
) -> tuple[float, float, int]:
    """Check true/false ports and inter-pipe overlap on the route masks."""
    margin, _, cell = _geometry(n, size)
    true_edges = {
        frozenset((tuple(first), tuple(second)))
        for segment in segments
        for first, second in zip(segment, segment[1:])
    }
    owner = {
        tuple(cell_rc): pipe_index
        for pipe_index, segment in enumerate(segments)
        for cell_rc in segment
    }
    pixels = [mask.load() for mask in masks]

    def bridge(mask_index: int, x: int, y: int, dr: int, dc: int) -> bool:
        px = pixels[mask_index]
        for tangent in (-1, 0, 1):
            centre = (x, y + tangent) if dc else (x + tangent, y)
            minus = (x - 1, y + tangent) if dc else (x + tangent, y - 1)
            plus = (x + 1, y + tangent) if dc else (x + tangent, y + 1)
            if px[centre] and px[minus] and px[plus]:
                return True
        return False

    true_total = true_hit = false_total = false_clear = 0
    for row in range(n):
        for column in range(n):
            first = (row, column)
            for dr, dc in ((0, 1), (1, 0)):
                second = row + dr, column + dc
                if not (0 <= second[0] < n and 0 <= second[1] < n):
                    continue
                x = int(round(margin + (column + 0.5 + dc / 2) * cell))
                y = int(round(margin + (row + 0.5 + dr / 2) * cell))
                if frozenset((first, second)) in true_edges:
                    true_total += 1
                    true_hit += bridge(owner[first], x, y, dr, dc)
                else:
                    false_total += 1
                    false_clear += not any(
                        bridge(pipe_index, x, y, dr, dc)
                        for pipe_index in range(len(masks))
                    )

    overlap_pixels = 0
    for index, first in enumerate(masks):
        for second in masks[index + 1:]:
            overlap_pixels += sum(ImageChops.multiply(first, second).getdata()) // 255
    return (
        true_hit / true_total if true_total else 1.0,
        false_clear / false_total if false_total else 1.0,
        overlap_pixels,
    )


def _draw_true_port_bridges_per_pipe(
    image: Image.Image,
    n: int,
    segments: Sequence[Sequence[Sequence[int]]],
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    size: int,
    styles: Sequence[PipeStyle],
    pipe_widths: Sequence[int],
) -> None:
    """Reopen true ports using the style assigned to each logical pipe."""
    if not (len(segments) == len(colors) == len(styles) == len(pipe_widths)):
        raise ValueError("per-pipe bridge inputs have different lengths")
    _, _, cell = _geometry(n, size)
    for segment, (_, rgb), style, pipe_width in zip(
        segments, colors, styles, pipe_widths
    ):
        bridge_width = min(pipe_width, max(3, int(round(cell * 0.30))))
        for first, second in zip(segment, segment[1:]):
            ax, ay = _center(first, n, size)
            bx, by = _center(second, n, size)
            midpoint = ((ax + bx) / 2, (ay + by) / 2)
            dx, dy = bx - ax, by - ay
            norm = max(abs(dx), abs(dy))
            ux, uy = dx / norm, dy / norm
            half = cell * 0.11
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.line(
                [
                    (midpoint[0] - ux * half, midpoint[1] - uy * half),
                    (midpoint[0] + ux * half, midpoint[1] + uy * half),
                ],
                fill=255,
                width=bridge_width,
            )
            _paste(image, rgb, mask, style.opacity_percent)
            core_fraction = style.core_fraction
            if style.texture == "center_rail" and not core_fraction:
                core_fraction = min(0.20, style.width_fraction * 0.28)
            if core_fraction:
                core_width = max(3, int(round(cell * core_fraction)))
                core_mask = Image.new("L", (size, size), 0)
                ImageDraw.Draw(core_mask).line(
                    [
                        (midpoint[0] - ux * half, midpoint[1] - uy * half),
                        (midpoint[0] + ux * half, midpoint[1] + uy * half),
                    ],
                    fill=255,
                    width=min(core_width, bridge_width),
                )
                _paste(image, rgb, core_mask, style.core_opacity_percent)


def style_knob_signature(style: PipeStyle) -> tuple[Any, ...]:
    """Return the visible knob tuple (everything but the style name and group)."""
    style.validate()
    return (
        round(style.width_fraction, 6),
        style.opacity_percent,
        style.texture,
        round(style.core_fraction, 6),
        style.core_opacity_percent,
        round(style.outline_fraction, 6),
        style.outline_opacity_percent,
        round(style.halo_fraction, 6),
        style.halo_opacity_percent,
        style.cell_fill_opacity_percent,
        style.grid_on_top,
        style.canonical_renderer,
    )


def render_pipe_styles(
    base: board_core.BaseBoard,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    appearance: board_render.Appearance,
    styles: Sequence[PipeStyle],
    size: int = board_core.SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one style per pipe and return the rendering metadata."""
    if len(styles) != len(base.segments):
        raise ValueError("exactly one PipeStyle is required per logical pipe")
    if len(colors) != len(base.segments):
        raise ValueError("exactly one colour is required per logical pipe")
    for style in styles:
        style.validate()
    board_core.validate_base(base.n, base.segments)
    _, _, cell = _geometry(base.n, size)
    route_masks = []
    pipe_widths = []
    for segment, style in zip(base.segments, styles):
        route_mask, pipe_width = _path_mask(
            base.n, segment, size, style.width_fraction
        )
        route_masks.append(route_mask)
        pipe_widths.append(pipe_width)
    true_recall, false_clearance, overlap_pixels = _mask_topology(
        base.n, base.segments, route_masks, size
    )
    if true_recall != 1.0 or false_clearance != 1.0 or overlap_pixels:
        raise ValueError(
            "unsafe route masks: "
            f"true={true_recall}, false={false_clearance}, overlap={overlap_pixels}"
        )

    canonical_count = sum(style.canonical_renderer for style in styles)
    if canonical_count and canonical_count != len(styles):
        raise ValueError("canonical renderer cannot be mixed with per-pipe styles")

    if canonical_count:
        if len({style_knob_signature(style) for style in styles}) != 1:
            raise ValueError("canonical renderer requires one board-shared style")
        image = board_render.draw_board(base.n, base.segments, colors, size, appearance)
        pipe_widths = [max(8, int(cell * 0.40))] * len(base.segments)
    else:
        image = board_render.textured_background(size, appearance)  # noqa: SLF001
        _draw_grid(image, base.n, size, appearance.grid_rgb)

        # A diffuse halo is a sleeve, never the only continuity signal.
        for segment, (_, rgb), style in zip(base.segments, colors, styles):
            if style.halo_fraction and style.halo_opacity_percent:
                halo_width_fraction = min(
                    0.88, style.width_fraction + style.halo_fraction
                )
                halo_mask, _ = _path_mask(
                    base.n, segment, size, halo_width_fraction
                )
                _paste(image, rgb, halo_mask, style.halo_opacity_percent)

        for segment, (_, rgb), style in zip(base.segments, colors, styles):
            if style.cell_fill_opacity_percent:
                _paste(
                    image,
                    rgb,
                    _cell_fill_mask(base.n, segment, size),
                    style.cell_fill_opacity_percent,
                )

        outline_rgb = _adaptive_outline(appearance.background_rgb)
        for segment, style in zip(base.segments, styles):
            if style.outline_fraction:
                outer_fraction = style.width_fraction + 2 * style.outline_fraction
                outer_mask, _ = _path_mask(
                    base.n, segment, size, outer_fraction
                )
                _paste(
                    image, outline_rgb, outer_mask, style.outline_opacity_percent
                )

        patterns = {
            texture: _pattern_mask(size, cell, texture)
            for texture in {style.texture for style in styles}
        }
        for segment, (_, rgb), route_mask, style in zip(
            base.segments, colors, route_masks, styles
        ):
            # First construct the full-colour texture, then apply the declared
            # outer opacity exactly once.  Sequential alpha composites would
            # make dots/stripes more opaque than their metadata claims.
            texture_layer = Image.new("RGB", (size, size), tuple(map(int, rgb)))
            if style.texture in {"dots", "diagonal_stripes", "crosshatch"}:
                clipped = ImageChops.multiply(route_mask, patterns[style.texture])
                texture_layer.paste(_accent(rgb), (0, 0), clipped)
            image.paste(
                texture_layer,
                (0, 0),
                _opacity_mask(route_mask, style.opacity_percent),
            )

            core_fraction = style.core_fraction
            if style.texture == "center_rail" and not core_fraction:
                core_fraction = min(0.20, style.width_fraction * 0.28)
            if core_fraction:
                # A narrow source-colour core protects continuity and colour
                # semantics for low-alpha outer ribbons.
                rail_mask, _ = _path_mask(
                    base.n,
                    segment,
                    size,
                    core_fraction,
                    endpoint_caps=False,
                )
                _paste(image, rgb, rail_mask, style.core_opacity_percent)

        if any(style.grid_on_top or style.width_fraction >= 0.82 for style in styles):
            _draw_grid(image, base.n, size, appearance.grid_rgb)
            _draw_true_port_bridges_per_pipe(
                image,
                base.n,
                base.segments,
                colors,
                size,
                styles,
                pipe_widths,
            )

    contrast_rows = []
    distances = []
    for pipe_index, ((name, rgb), style) in enumerate(zip(colors, styles)):
        alpha = style.opacity_percent / 100
        visible = _mix(appearance.background_rgb, rgb, alpha)
        distance = math.dist(
            pipe_colors.lab(appearance.background_rgb), pipe_colors.lab(visible)
        )
        distances.append(distance)
        contrast_rows.append({
            "pipe_index": pipe_index,
            "color_name": name,
            "source_rgb": list(rgb),
            "flat_background_composite_rgb": list(visible),
            "background_de": round(distance, 2),
        })
    signatures = [style_knob_signature(style) for style in styles]
    shared = len(set(signatures)) == 1
    low_alpha_indices = [
        index
        for index, style in enumerate(styles)
        if (
            style.opacity_percent < 80
            and not (
                style.core_fraction
                and style.core_opacity_percent >= 80
            )
            and not (
                style.texture == "center_rail"
                and style.core_opacity_percent >= 80
            )
        )
    ]
    meta = {
        "schema_version": VERSION,
        "style_scope": "board_shared" if shared else "per_pipe",
        "style": dataclasses.asdict(styles[0]) if shared else None,
        "pipe_styles": [dataclasses.asdict(style) for style in styles],
        "grid_n": base.n,
        "image_size": size,
        "cell_px": round(cell, 3),
        "pipe_width_px": pipe_widths[0] if shared else None,
        "pipe_width_px_by_pipe": pipe_widths,
        "actual_pipe_width_fraction": (
            round(pipe_widths[0] / cell, 4) if shared else None
        ),
        "actual_pipe_width_fraction_by_pipe": [
            round(width / cell, 4) for width in pipe_widths
        ],
        "route_mask_true_port_recall": round(true_recall, 6),
        "route_mask_false_port_clearance": round(false_clearance, 6),
        "route_mask_pairwise_overlap_pixels": overlap_pixels,
        "min_flat_background_pipe_de": round(min(distances), 2),
        "flat_background_contrast_by_color": contrast_rows,
        "low_alpha_requires_opaque_core_or_outline": bool(low_alpha_indices),
        "low_alpha_requires_source_color_core": bool(low_alpha_indices),
        "low_alpha_unguarded_pipe_indices": low_alpha_indices,
        "area_shortcut_risk": any(
            bool(style.cell_fill_opacity_percent) for style in styles
        ),
        "pattern_scope": "global_canvas_shared_by_all_pipes",
        "pattern_punches_holes": False,
    }
    return image.convert("RGB"), meta


__all__ = [
    "PipeStyle",
    "TEXTURES",
    "VERSION",
    "render_pipe_styles",
    "style_knob_signature",
]
