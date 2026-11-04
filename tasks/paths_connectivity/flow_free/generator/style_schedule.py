"""Deterministic material support for Flow Free.

The three visible knobs are sampled independently over a discrete support,
subject only to the explicit compositional holdout.  Easy through xhard never
combine a low-alpha sleeve (50--70%) with a texture, nor a near-fill width with
low alpha.  Those joint regions are reserved for xxhard OOD while every width,
alpha, and texture marginal is already present in ID.

Half of ID boards share one material across every pipe.  The other half sample
once per distinct colour group.  The deliberate same-colour target/distractor
pair therefore always receives the same material and cannot be identified by
width, alpha, or texture.
"""
from __future__ import annotations
from visro.seeding import seed_text

import dataclasses
import hashlib
import json
import random
from typing import Any, Sequence

from . import board_core, pipe_styles

RNG_NAMESPACE = seed_text(46327499350614196167315201799821678051958810777429489530077871087862073876273)  # fixed RNG namespace (_rng)
WIDTHS = (0.28, 0.40, 0.56, 0.64, 0.70, 0.76, 0.86)
ALPHAS = (50, 60, 70, 85, 100)
TEXTURES = ("solid", "dots", "diagonal_stripes", "crosshatch", "center_rail")
ASSIGNMENT_MODES = ("board_shared", "per_pipe_independent")


def is_compositional_ood_values(width: float, alpha: int, texture: str) -> bool:
    """Return whether a knob tuple lies in the held-out joint region."""
    return (
        (alpha <= 70 and texture != "solid")
        or (width >= 0.78 and alpha <= 70)
    )


def _style(
    domain: str, width: float, alpha: int, texture: str
) -> pipe_styles.PipeStyle:
    low_alpha = alpha < 80
    core = 0.0
    if low_alpha:
        core = 0.14 if width <= 0.40 else (0.16 if width <= 0.56 else 0.18)
    name = (
        f"{domain.lower()}_w{round(width * 100):02d}_"
        f"a{alpha:03d}_{texture}"
    )
    return pipe_styles.PipeStyle(
        name=name,
        group=domain,
        width_fraction=width,
        opacity_percent=alpha,
        texture=texture,
        core_fraction=core,
        core_opacity_percent=100,
        grid_on_top=width >= 0.82,
    )


# This is the full independently crossed support after removing the held-out
# region.  The restriction intentionally changes the conditional distribution:
# low-alpha ID pipes are solid and carry an opaque source-colour core.
ID_STYLE_BANK = tuple(
    _style("ID_SUPPORT", width, alpha, texture)
    for width in WIDTHS
    for alpha in ALPHAS
    for texture in TEXTURES
    if not is_compositional_ood_values(width, alpha, texture)
)

# A compact, balanced subset of the held-out region is used in OOD.  It spans
# every marginal exposed in ID without mixing unsafe near-fill waves.
_OOD_VALUES = (
    (0.86, 50, "dots"),
    (0.76, 60, "diagonal_stripes"),
    (0.70, 70, "crosshatch"),
    (0.64, 50, "center_rail"),
    (0.56, 60, "dots"),
    (0.40, 70, "diagonal_stripes"),
    (0.28, 50, "crosshatch"),
    (0.86, 70, "center_rail"),
)
OOD_STYLE_BANK = tuple(
    _style("OOD_SUPPORT", width, alpha, texture)
    for width, alpha, texture in _OOD_VALUES
)


def style_signature(style: pipe_styles.PipeStyle) -> tuple[Any, ...]:
    return pipe_styles.style_knob_signature(style)


def support_sha256(styles: Sequence[pipe_styles.PipeStyle]) -> str:
    payload = [list(style_signature(style)) for style in styles]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


ID_SUPPORT_SHA256 = support_sha256(ID_STYLE_BANK)
OOD_SUPPORT_SHA256 = support_sha256(OOD_STYLE_BANK)


def validate_supports() -> None:
    if len(ID_STYLE_BANK) != len({style_signature(style) for style in ID_STYLE_BANK}):
        raise ValueError("duplicate ID material tuple")
    if len(OOD_STYLE_BANK) != len({style_signature(style) for style in OOD_STYLE_BANK}):
        raise ValueError("duplicate OOD material tuple")
    for style in ID_STYLE_BANK + OOD_STYLE_BANK:
        style.validate()
    if any(
        is_compositional_ood_values(
            style.width_fraction, style.opacity_percent, style.texture
        )
        for style in ID_STYLE_BANK
    ):
        raise ValueError("ID material entered the held-out region")
    if not all(
        is_compositional_ood_values(
            style.width_fraction, style.opacity_percent, style.texture
        )
        for style in OOD_STYLE_BANK
    ):
        raise ValueError("OOD material left the held-out region")
    id_widths = {style.width_fraction for style in ID_STYLE_BANK}
    id_alphas = {style.opacity_percent for style in ID_STYLE_BANK}
    id_textures = {style.texture for style in ID_STYLE_BANK}
    for style in OOD_STYLE_BANK:
        if style.width_fraction not in id_widths:
            raise ValueError("OOD introduced a new width marginal")
        if style.opacity_percent not in id_alphas:
            raise ValueError("OOD introduced a new alpha marginal")
        if style.texture not in id_textures:
            raise ValueError("OOD introduced a new texture marginal")


validate_supports()


def _rng(*parts: Any) -> random.Random:
    raw = "|".join(map(str, (RNG_NAMESPACE,) + parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:16], "big"))


def _colour_key(color: tuple[str, tuple[int, int, int]]) -> str:
    name, rgb = color
    return f"{name}:{rgb[0]},{rgb[1]},{rgb[2]}"


def id_assignment_mode(spec: board_core.Spec) -> str:
    """A 50/50 factor crossed with tier, form, layout, and nuisance row."""
    if spec.dataset_role in board_core.HOLDOUT_ID_ROLES:
        # The record-level Latin schedule fixes tier by slot and rotates the
        # question form with block+slot. Mixing two block scales with the form
        # index gives eval an exact 75/75 split inside every tier, validation a
        # nearest-possible 12/13 split inside every tier, and at most one item
        # of imbalance in every tier x form x layout cell.
        form_index = board_core.QUESTION_FORMS.index(spec.question_form)
        value = spec.block // 4 + spec.block // 8 + form_index
    else:
        # ``block`` alone correlates with the train tier cycle, while
        # ``nuisance_row`` alone is identical to layout parity.  Toggling each
        # complete 128-record cycle makes assignment scope 1562/1563 inside
        # every 100k train tier x form x layout cell.
        value = spec.block + spec.nuisance_row + spec.block // 32
    return ASSIGNMENT_MODES[value % 2]


def _sample_per_colour(
    *,
    bank: Sequence[pipe_styles.PipeStyle],
    seed: int,
    split: str,
    index: int,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    domain: str,
) -> tuple[list[pipe_styles.PipeStyle], dict[str, Any]]:
    group_keys = sorted({_colour_key(color) for color in colors})
    minimum_variety = min(2, len(group_keys))
    for nonce in range(128):
        by_group = {}
        for key in group_keys:
            rng = _rng(seed, split, index, domain, key, nonce)
            by_group[key] = bank[rng.randrange(len(bank))]
        chosen = list(by_group.values())
        if (
            len({style.width_fraction for style in chosen}) >= minimum_variety
            and len({style.opacity_percent for style in chosen}) >= minimum_variety
            and len({style.texture for style in chosen}) >= minimum_variety
        ):
            styles = [by_group[_colour_key(color)] for color in colors]
            return styles, {
                "sampling_nonce": nonce,
                "style_group_count": len(group_keys),
                "independent_group_keys": group_keys,
                "same_color_style_tied": True,
            }
    raise RuntimeError("could not sample a diverse per-colour material assignment")


def assign_id_styles(
    seed: int,
    split: str,
    index: int,
    spec: board_core.Spec,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
) -> tuple[str, list[pipe_styles.PipeStyle], dict[str, Any]]:
    mode = id_assignment_mode(spec)
    if mode == "board_shared":
        rng = _rng(seed, split, index, "id_board_shared")
        style = ID_STYLE_BANK[rng.randrange(len(ID_STYLE_BANK))]
        return mode, [style] * len(colors), {
            "sampling_nonce": None,
            "style_group_count": 1,
            "independent_group_keys": [],
            "same_color_style_tied": True,
        }
    styles, sampling = _sample_per_colour(
        bank=ID_STYLE_BANK,
        seed=seed,
        split=split,
        index=index,
        colors=colors,
        domain="id_per_pipe",
    )
    return mode, styles, sampling


def assign_ood_styles(
    seed: int,
    index: int,
    mode: str,
    colors: Sequence[tuple[str, tuple[int, int, int]]],
) -> tuple[list[pipe_styles.PipeStyle], dict[str, Any]]:
    if mode == "board_shared":
        # There are exactly 120 records in this production stratum, so every
        # curated OOD combination occurs 15 times.
        style = OOD_STYLE_BANK[(index // 5) % len(OOD_STYLE_BANK)]
        return [style] * len(colors), {
            "sampling_nonce": None,
            "style_group_count": 1,
            "independent_group_keys": [],
            "same_color_style_tied": True,
        }
    if mode != "per_pipe_independent":
        raise ValueError(f"unknown OOD material assignment mode: {mode}")
    return _sample_per_colour(
        bank=OOD_STYLE_BANK,
        seed=seed,
        split="ood",
        index=index,
        colors=colors,
        domain="ood_per_pipe",
    )


def assignment_rows(
    colors: Sequence[tuple[str, tuple[int, int, int]]],
    styles: Sequence[pipe_styles.PipeStyle],
    realized_widths: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "pipe_index": pipe_index,
            "color_name": name,
            "color_rgb": list(rgb),
            "style_group_key": _colour_key((name, rgb)),
            "combination_id": style.name,
            "knob_signature": list(style_signature(style)),
            "style": dataclasses.asdict(style),
            "realized_width_px": int(realized_widths[pipe_index]),
        }
        for pipe_index, ((name, rgb), style) in enumerate(zip(colors, styles))
    ]


__all__ = [
    "ALPHAS",
    "ASSIGNMENT_MODES",
    "ID_STYLE_BANK",
    "ID_SUPPORT_SHA256",
    "OOD_STYLE_BANK",
    "OOD_SUPPORT_SHA256",
    "TEXTURES",
    "RNG_NAMESPACE",
    "WIDTHS",
    "assign_id_styles",
    "assign_ood_styles",
    "assignment_rows",
    "id_assignment_mode",
    "is_compositional_ood_values",
    "style_signature",
    "support_sha256",
    "validate_supports",
]
