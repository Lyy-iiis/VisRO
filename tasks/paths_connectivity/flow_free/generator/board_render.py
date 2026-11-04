"""Question-independent appearance augmentation for Flow Free.

Only the background surface and grid ink vary in the in-distribution training
profile.  The themes deliberately span dark, warm-light, cool-light, and
green-light surfaces instead of four nearly black variants.  Texture is always
composited before the grid, pipes, and endpoint annotations, so it cannot hide
or alter the latent path graph.
"""
from __future__ import annotations
from visro.seeding import seed_text

import dataclasses
import hashlib
import math
import random
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclasses.dataclass(frozen=True)
class Appearance:
    family: str
    background_rgb: tuple[int, int, int]
    grid_rgb: tuple[int, int, int]
    texture_style: str
    texture_amplitude: int
    texture_scale_px: int
    texture_seed: int
    pipe_style: str = "orthogonal_solid"


THEMES = (
    # Every base background is at least dE 47 from every named pipe colour.
    # Grid inks are deliberately not the base generator's red and remain at least
    # dE 36 from every pipe colour, even after the bounded per-example jitter.
    ("midnight_slate", (32, 36, 44), (104, 104, 128)),
    ("warm_parchment", (240, 224, 192), (152, 152, 152)),
    ("mist_lavender", (216, 192, 232), (128, 120, 136)),
    ("sage_paper", (192, 208, 144), (136, 144, 128)),
)
RGB_JITTER = 5

# Appearance is an independent block factor, not a function of the nuisance
# row.  This matters because nuisance rows carry label/ascending/descending
# and circle/colour wording.  The train table is a balanced colouring of the
# 32-block tier x nuisance-row supercycle; its first 16 blocks are balanced too
# for any 64-record prefix.  The OOD-control table similarly colours
# four occurrences of every nuisance row in its 32-block cycle.
TRAIN_THEME_BY_BLOCK = (
    0, 3, 1, 2, 1, 2, 0, 3,
    2, 0, 2, 1, 3, 1, 3, 0,
    0, 2, 1, 3, 2, 0, 3, 1,
    3, 1, 2, 0, 1, 3, 0, 2,
)
OOD_THEME_BY_BLOCK = (
    2, 3, 0, 2, 3, 2, 3, 0,
    0, 1, 1, 3, 1, 0, 2, 1,
    1, 0, 2, 0, 0, 1, 0, 2,
    3, 2, 3, 1, 2, 3, 1, 3,
)
ID_DATASET_ROLES = frozenset(("train_id", "eval_id", "val_id"))
TEXTURE_BY_THEME = (
    ("plain", 0, 0),
    ("mottle", 5, 64),
    ("plain", 0, 0),
    ("micrograin", 3, 4),
)
assert len(TRAIN_THEME_BY_BLOCK) == len(OOD_THEME_BY_BLOCK) == 32


_APPEARANCE_NAMESPACE = seed_text(642923205581410772467312228718378726607002532433143773080882)  # fixed RNG namespace


def _stable_seed(seed: int | str, block: int, scope: str) -> int:
    raw = f"{_APPEARANCE_NAMESPACE}|{seed}|{block}|{scope}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def appearance_for(seed: int | str, spec: Any) -> Appearance:
    """Return a paired appearance: length/turn share it within each scope."""
    table = (
        TRAIN_THEME_BY_BLOCK
        if spec.dataset_role in ID_DATASET_ROLES else OOD_THEME_BY_BLOCK
    )
    supercycle, position = divmod(spec.block, len(table))
    theme_index = (table[position] + supercycle) % len(THEMES)
    name, background, grid = THEMES[theme_index]
    texture, amplitude, scale = TEXTURE_BY_THEME[theme_index]
    colour_rng = random.Random(_stable_seed(seed, spec.block, spec.scope))

    def jitter(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            max(0, min(255, value + colour_rng.randint(-RGB_JITTER, RGB_JITTER)))
            for value in rgb
        )

    return Appearance(
        family=f"id_augmented/{name}/{texture}",
        background_rgb=jitter(background),
        grid_rgb=jitter(grid),
        texture_style=texture,
        texture_amplitude=amplitude,
        texture_scale_px=scale,
        texture_seed=_stable_seed(seed, spec.block, spec.scope),
    )


def textured_background(size: int, appearance: Appearance) -> Image.Image:
    bg = appearance.background_rgb
    if appearance.texture_style == "plain":
        return Image.new("RGB", (size, size), bg)
    rng = random.Random(appearance.texture_seed)
    if appearance.texture_style == "mottle":
        low = max(4, math.ceil(size / appearance.texture_scale_px))
        values = []
        for _ in range(low * low):
            delta = rng.randint(-appearance.texture_amplitude, appearance.texture_amplitude)
            values.append(tuple(max(0, min(255, value + delta)) for value in bg))
        layer = Image.new("RGB", (low, low))
        layer.putdata(values)
        # Bilinear interpolation stays inside the declared +/- amplitude;
        # bicubic can overshoot the control-point range.
        return layer.resize((size, size), Image.Resampling.BILINEAR)
    if appearance.texture_style == "micrograin":
        low = max(8, math.ceil(size / appearance.texture_scale_px))
        values = []
        for _ in range(low * low):
            delta = rng.randint(-appearance.texture_amplitude, appearance.texture_amplitude)
            values.append(tuple(max(0, min(255, value + delta)) for value in bg))
        layer = Image.new("RGB", (low, low))
        layer.putdata(values)
        return layer.resize((size, size), Image.Resampling.BILINEAR)
    raise ValueError(f"unknown texture style: {appearance.texture_style}")


def draw_board(
    n: int,
    segments: Sequence[Sequence[Sequence[int]]],
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    size: int,
    appearance: Appearance,
) -> Image.Image:
    """Render the stock orthogonal pipe grammar over a safe augmented surface."""
    image = textured_background(size, appearance)
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.05)
    board = size - 2 * margin
    cell = board / n

    def center(rc: Sequence[int]) -> tuple[float, float]:
        row, column = int(rc[0]), int(rc[1])
        return margin + (column + 0.5) * cell, margin + (row + 0.5) * cell

    for i in range(n + 1):
        x = margin + i * cell
        y = margin + i * cell
        draw.line([(x, margin), (x, margin + board)], fill=appearance.grid_rgb, width=1)
        draw.line([(margin, y), (margin + board, y)], fill=appearance.grid_rgb, width=1)

    pipe_width = max(8, int(cell * 0.40))
    cap = int(cell * 0.34)
    for segment, (_, rgb) in zip(segments, colors):
        points = [center(cell_rc) for cell_rc in segment]
        for first, second in zip(points, points[1:]):
            draw.line([first, second], fill=rgb, width=pipe_width)
        for x, y in points:
            radius = pipe_width / 2
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=rgb)
        for x, y in (points[0], points[-1]):
            draw.rectangle([x - cap, y - cap, x + cap, y + cap], fill=rgb)
            draw.rectangle(
                [x - cap + 3, y - cap + 3, x + cap - 3, y + cap - 3],
                outline=tuple(int(value * 0.6) for value in rgb),
                width=2,
            )
    return image


def metadata(appearance: Appearance, colors: Sequence[tuple[str, tuple[int, int, int]]], lab) -> dict[str, Any]:
    background_lab = lab(appearance.background_rgb)
    min_pipe_background = min(
        math.dist(background_lab, lab(rgb)) for _, rgb in colors
    )
    grid_background = math.dist(background_lab, lab(appearance.grid_rgb))
    min_pipe_grid = min(
        math.dist(lab(appearance.grid_rgb), lab(rgb)) for _, rgb in colors
    )
    if min_pipe_background < 40:
        raise ValueError(f"unsafe pipe/background contrast: dE={min_pipe_background:.2f}")
    if grid_background < 18:
        raise ValueError(f"unsafe grid/background contrast: dE={grid_background:.2f}")
    if min_pipe_grid < 30:
        raise ValueError(f"unsafe pipe/grid contrast: dE={min_pipe_grid:.2f}")
    return {
        "appearance_family": appearance.family,
        "background_rgb": list(appearance.background_rgb),
        "grid_rgb": list(appearance.grid_rgb),
        "texture_style": appearance.texture_style,
        "texture_amplitude_rgb": appearance.texture_amplitude,
        "texture_scale_px": appearance.texture_scale_px,
        "texture_seed": appearance.texture_seed,
        "rgb_jitter_bound": RGB_JITTER,
        "pipe_style": appearance.pipe_style,
        "compositing_order": ["background", "texture", "grid", "pipes", "endpoint_annotations"],
        "min_pipe_background_de": round(min_pipe_background, 2),
        "grid_background_de": round(grid_background, 2),
        "min_pipe_grid_de": round(min_pipe_grid, 2),
    }
