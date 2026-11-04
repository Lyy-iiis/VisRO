"""Mask and tile primitives of find_the_shadow: an icon's render, its alpha
silhouette, and the colours and paste rule of the grid tiles."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

BG = (255, 255, 255)

# vivid tints for silhouette "specific" tiles (so they look like real objects)
SPECIFIC_TINTS = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 64, 122),
    (41, 128, 185), (39, 174, 96),
]
# dark single-color shadow fills
SHADOW_COLORS = [(45, 45, 55), (35, 40, 70), (50, 50, 50), (30, 45, 60)]


# --------------------------------------------------------------------------- masks
def alpha_mask(img: Image.Image, thresh: int = 40) -> np.ndarray:
    """Boolean foreground mask from an RGBA tile's alpha channel."""
    a = np.asarray(img.convert("RGBA"))
    return a[..., 3] > thresh


def tile_silhouette(img: Image.Image, S: int) -> np.ndarray:
    """Silhouette of an RGBA tile as the largest connected opaque component,
    resampled to SxS."""
    small = img.resize((S, S), Image.LANCZOS)
    m = alpha_mask(small).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def render_canon(lib, entry, S, tint):
    """Render an icon to RGBA at SxS; for silhouettes tint to `tint` (else own colours)."""
    color = tint if entry["style"] == "silhouette" else None
    return lib.render(entry, S, color=color)


# --------------------------------------------------------------------------- draw
def fit_tile(canvas, box, tile_rgba):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    s = min(w, h)
    sub = tile_rgba.resize((s, s), Image.LANCZOS)
    canvas.paste(sub, (x0 + (w - s) // 2, y0 + (h - s) // 2), sub)
