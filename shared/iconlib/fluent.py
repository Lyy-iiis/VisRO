"""Semantic Fluent Emoji assets, separate from the SVG icon library (iconlib).

An identity (for example red apple) has several styles, and several identities
can share a counting class (apple). Sampling weights classes, then identities,
then styles uniformly. Rendering preserves the original RGBA, including shading.
Rotation, scaling, and mirroring are image transforms, not new 3D poses.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text


# schema label stored in the pinned fluent_manifest.json / fluent_catalog.json
SCHEMA_VERSION = seed_text(164588486048695051706877160126359422275682418761603546363033137)
STYLES = frozenset({"3d", "color", "flat", "high_contrast"})
ALPHA_THRESHOLD = 40


class FluentLibrary:
    """Load a pinned manifest of prepared, full-color RGBA PNG variants."""

    def __init__(self, manifest_path, asset_root=None):
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected manifest schema {SCHEMA_VERSION!r}")
        self.source = manifest["source"]
        root = Path(asset_root if asset_root is not None else manifest["asset_root"])
        self.root = (
            root.resolve()
            if root.is_absolute() or asset_root is not None
            else ((_ROOT / 'assets/fluent_countable') if self.manifest_path == (_ROOT / 'shared/iconlib/fluent_manifest.json') else self.manifest_path.parent / root).resolve()
        )
        self.entries = manifest["entries"]
        self._assets = []
        seen_identities = set()
        class_splits = {}
        group_splits = {}
        for entry in self.entries:
            identity = entry["identity_id"]
            class_id = entry["class_id"]
            split = entry["split"]
            self._check_split(split)
            if split is None:
                raise ValueError(f"missing split for {identity!r}")
            if identity in seen_identities:
                raise ValueError(f"duplicate identity_id: {identity!r}")
            seen_identities.add(identity)
            for label, value, assignments in (
                ("class_id", class_id, class_splits),
                ("split_group", entry["split_group"], group_splits),
            ):
                if value in assignments and assignments[value] != split:
                    raise ValueError(f"{label} {value!r} appears in multiple splits")
                assignments[value] = split
            if not entry["variants"]:
                raise ValueError(f"identity {identity!r} has no variants")
            seen_styles = set()
            for variant in entry["variants"]:
                style = variant["style"]
                if style not in STYLES:
                    raise ValueError(f"unknown Fluent style: {style!r}")
                if style in seen_styles:
                    raise ValueError(f"duplicate style {style!r} for {identity!r}")
                seen_styles.add(style)
                path = Path(variant["path"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"asset path must stay inside asset_root: {path}")
                self._assets.append({
                    **{key: value for key, value in entry.items() if key != "variants"},
                    **variant,
                    "asset_id": f"{identity}/{style}",
                    "source": dict(self.source),
                })
        self._assets.sort(key=lambda asset: asset["asset_id"])

    @staticmethod
    def _check_split(split):
        if split not in (None, "train", "eval", "val", "test", "catalog"):
            raise ValueError(f"unknown split: {split!r}")

    @staticmethod
    def _styles(styles):
        if styles is None:
            return STYLES
        selected = frozenset([styles] if isinstance(styles, str) else styles)
        if not selected or not selected.issubset(STYLES):
            raise ValueError(f"expected nonempty Fluent styles from {sorted(STYLES)}")
        return selected

    def classes(self, split=None):
        """Return sorted semantic class IDs, optionally from one split."""
        self._check_split(split)
        return sorted({
            asset["class_id"] for asset in self._assets
            if split is None or asset["split"] == split
        })

    def _eligible(self, class_id, styles, split):
        self._check_split(split)
        selected_styles = self._styles(styles)
        assets = [
            asset for asset in self._assets
            if (class_id is None or asset["class_id"] == class_id)
            and (split is None or asset["split"] == split)
            and asset["style"] in selected_styles
        ]
        if not assets:
            raise ValueError(
                f"no Fluent assets for class_id={class_id!r}, "
                f"styles={sorted(selected_styles)!r}, split={split!r}"
            )
        return assets

    def variants(self, class_id, styles=None, split=None):
        """Return flat metadata records for available variants of a class.

        Style filters select their available intersection; an empty result
        raises ValueError. Each record includes provenance and identity/class
        fields, so the caller can count classes while recording exact assets.
        """
        return [dict(asset) for asset in self._eligible(class_id, styles, split)]

    def pick(self, rng, class_id=None, styles=None, split="train"):
        """Pick uniformly by eligible class, identity, and then style.

        Pass split=None only when deliberately sampling all classes. ``eval``
        is the held-out pool of the eval and ood items; ``catalog`` is the full
        inventory. Reusing a seeded random.Random with the same manifest gives
        reproducible picks.
        """
        assets = self._eligible(class_id, styles, split)
        classes = sorted({asset["class_id"] for asset in assets})
        chosen_class = class_id if class_id is not None else rng.choice(classes)
        assets = [asset for asset in assets if asset["class_id"] == chosen_class]
        identities = sorted({asset["identity_id"] for asset in assets})
        chosen_identity = rng.choice(identities)
        assets = [asset for asset in assets if asset["identity_id"] == chosen_identity]
        return dict(rng.choice(assets))

    def render(self, asset, size=128, angle=0, scale=1, mirror=False):
        """Render the original RGBA with a normalized foreground size and no clipping.

        ``size * scale`` is the longest foreground-bounding-box edge before
        rotation. Alpha > 40 defines this box and its centroid; all nonzero
        source alpha is retained, including antialiasing outside that box.
        This threshold does not claim to separate objects from cast shadows.

        The returned canvas is centered on the foreground centroid, padded,
        and expanded for rotation. Its dimensions can exceed ``size``. The
        caller records transforms alongside the selected asset metadata.
        """
        if not all(math.isfinite(value) for value in (size, scale, angle)):
            raise ValueError("size, scale, and angle must be finite")
        if size <= 0 or scale <= 0:
            raise ValueError("size and scale must be positive")
        path = (self.root / asset["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("asset path must stay inside asset_root")
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode != "RGBA":
                raise ValueError(f"prepared Fluent asset must be an RGBA PNG: {path}")
            sprite = source.copy()
        alpha = np.asarray(sprite.getchannel("A"))
        ys, xs = np.where(alpha > ALPHA_THRESHOLD)
        if len(xs) == 0:
            raise ValueError(f"Fluent asset has no visible foreground: {path}")
        factor = float(size) * float(scale) / max(
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        )
        support = sprite.getchannel("A").getbbox()
        sprite = sprite.crop(support)
        width = max(1, round(sprite.width * factor))
        height = max(1, round(sprite.height * factor))
        # Convert the source foreground centroid to the resized pixel grid.
        center_x = (float(xs.mean()) + 0.5 - support[0]) * width / sprite.width
        center_y = (float(ys.mean()) + 0.5 - support[1]) * height / sprite.height
        sprite = sprite.resize((width, height), Image.Resampling.LANCZOS)
        if mirror:
            sprite = ImageOps.mirror(sprite)
            center_x = width - center_x
        padding = 3
        canvas = Image.new("RGBA", (
            math.ceil(2 * max(center_x, width - center_x)) + 2 * padding,
            math.ceil(2 * max(center_y, height - center_y)) + 2 * padding,
        ))
        canvas.alpha_composite(sprite, (
            round(canvas.width / 2 - center_x),
            round(canvas.height / 2 - center_y),
        ))
        return canvas.rotate(
            angle, expand=True, resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
