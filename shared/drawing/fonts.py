"""The bundled DejaVu fonts (assets/fonts), loaded by name.

Generators load their label fonts from the repository rather than from the
operating system, so the same Pillow and FreeType versions render the same
pixels on every machine.
"""
from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

from visro.paths import ROOT

FONT_DIR = ROOT / 'assets/fonts'

# logical name -> filename in FONT_DIR
FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
}


def font_path(name: str = "sans") -> Path:
    return FONT_DIR / FONTS[name]


def load_bundled_font(size: int, name: str = "sans") -> ImageFont.FreeTypeFont:
    """Load a bundled TTF at `size`; raises FileNotFoundError if it is missing (no fallback to OS fonts)."""
    p = font_path(name)
    if not p.exists():
        raise FileNotFoundError(f"bundled font missing: {p}; restore assets/fonts/ in the VisRO repository")
    return ImageFont.truetype(str(p), size=size)
