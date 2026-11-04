"""Glyph stroke masks: a glyph re-weighted at a fixed font size, preserving components and counters."""

from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

FONT = _ROOT / 'assets/fonts/DejaVuSans-Bold.ttf'
FONT_SHA256 = 'b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167'


def _topology(binary):
    components = ndi.label(binary)[1]
    holes = ndi.label(ndi.binary_fill_holes(binary) & ~binary)[1]
    return components, holes


def make_stroke_mask(char, font_px, stroke_scale=1.0):
    """Change stroke width at fixed font size, preserving components and counters."""
    if not math.isfinite(stroke_scale) or not 0.6 <= stroke_scale <= 1.4:
        raise ValueError('stroke_scale must be within [0.6, 1.4]')
    if font_px < 16:
        raise ValueError('font_px must be at least 16')
    if not FONT.is_file() or hashlib.sha256(FONT.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('bundled DejaVuSans-Bold.ttf is missing or differs from its pinned hash')
    scale = 3
    font = ImageFont.truetype(str(FONT), round(font_px * scale))
    box = font.getbbox(char)
    pad = math.ceil(font_px * 0.20) * scale
    im = Image.new('L', (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad))
    ImageDraw.Draw(im).text((pad - box[0], pad - box[1]), char, font=font, fill=255)
    base = np.asarray(im) > 127
    inside = ndi.distance_transform_edt(base)
    outside = ndi.distance_transform_edt(~base)
    # Half-pixel correction places the signed boundary between opposite pixels.
    sdf = np.where(base, inside - 0.5, -outside + 0.5)
    ridge = base & (inside >= ndi.maximum_filter(inside, size=5) - 1e-8)
    radius = float(np.median(inside[ridge]))
    offset = (stroke_scale - 1) * radius
    effective = float(stroke_scale)
    original_topology = _topology(base)
    for _ in range(16):
        new = sdf + offset > 0
        if _topology(new) == original_topology and new.any():
            break
        offset *= 0.5
        effective = 1 + offset / radius
    alpha = np.clip((sdf + offset + 0.5) * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(alpha).resize(
        (round(im.width / scale), round(im.height / scale)), Image.Resampling.LANCZOS
    )
    # Check topology again at actual source resolution, where thin strokes matter.
    reference = im.resize(result.size, Image.Resampling.LANCZOS)
    if _topology(np.asarray(result) > 127) != _topology(np.asarray(reference) > 127):
        result = reference
        effective = 1.0
        offset = 0.0
    return result, dict(
        stroke_scale_requested=float(stroke_scale),
        stroke_scale_effective=float(effective),
        stroke_offset_px=float(offset / scale),
        base_stroke_width_px=float(2 * radius / scale),
        font_px=int(font_px),
        topology=list(original_topology),
    )
