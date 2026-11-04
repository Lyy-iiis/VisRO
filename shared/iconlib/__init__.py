"""Shared open-icon library for the VisRO generators.

Wraps ~16k real icons (Material Design Icons, game-icons.net, OpenMoji) behind a
small API:

    lib = IconLibrary()                       # loads svg_manifest.json
    e   = lib.pick(rng, style="silhouette")   # reproducible pick
    img = lib.render(e, S, color=(30,30,30))  # RGBA SxS (tinted if silhouette)
    alt, m, desc = lib.alter(e, img, rng, strength, color=...)  # one subtle change

`alter` returns a NEW RGBA tile that differs from `img` by exactly one localized
change that genuinely alters the icon: a part removed (a real SVG
sub-part for color icons, or a circle/square/triangle/line erase), a protrusion
added at the edge, a rotation, a mirror, or a recolor/hue-shift. Pure pose changes
(translation, uniform scale) are deliberately excluded -- they leave the shape
unchanged and are not valid "different" cues. It is the engine behind "find the
different" and supplies distractors for "find the same" / "find the shadow".

Rendering uses cairosvg. Everything is a pure function of the
caller's RNG + the pinned manifest, so problems stay reproducible.
"""
from __future__ import annotations
import io, json, math, re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from visro.paths import ROOT as _ROOT

try:
    import cairosvg
    HAVE_CAIROSVG = True
except ImportError:                       # some environments have no cairosvg
    cairosvg = None
    HAVE_CAIROSVG = False

MANIFEST = (_ROOT / 'shared/iconlib/svg_manifest.json')
SVG_NS = "http://www.w3.org/2000/svg"
DRAW = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}
ET.register_namespace("", SVG_NS)


def _ln(tag: str) -> str:
    return tag.split("}")[-1]


def _sanitize_broken_markers(svg: bytes) -> bytes:
    """Remove marker references whose target is absent from the SVG.

    Some OpenMoji files contain ``marker-end="url(#Dot)"`` without a
    corresponding ``<marker id="Dot">`` definition. CairoSVG dereferences that
    missing marker as ``None`` and crashes. Browsers simply omit the missing
    marker, so doing the same here is the least surprising recovery.
    """
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg

    marker_ids = {
        element.get("id")
        for element in root.iter()
        if _ln(element.tag) == "marker" and element.get("id")
    }
    changed = False
    for element in root.iter():
        for attr in ("marker", "marker-start", "marker-mid", "marker-end"):
            value = element.get(attr)
            if not value:
                continue
            match = re.fullmatch(r"url\(\s*#([^\s)]+)\s*\)", value)
            if match and match.group(1) not in marker_ids:
                del element.attrib[attr]
                changed = True
    return ET.tostring(root, encoding="utf-8") if changed else svg


def _svg2png(*, path: Path | None = None, svg: bytes | None = None,
             width: int, height: int) -> bytes:
    """Rasterize SVG bytes after tolerating broken external marker references."""
    if (path is None) == (svg is None):
        raise ValueError("pass exactly one of path or svg")
    source = path.read_bytes() if path is not None else svg
    return cairosvg.svg2png(
        bytestring=_sanitize_broken_markers(source),
        output_width=width,
        output_height=height,
        background_color=None,
    )


def _tint(img: Image.Image, rgb) -> Image.Image:
    """Recolor every opaque pixel of a silhouette to `rgb`, keeping alpha."""
    a = np.array(img)
    m = a[..., 3] > 0
    a[m, 0], a[m, 1], a[m, 2] = rgb
    return Image.fromarray(a, "RGBA")


def _fg_pixels(img: Image.Image):
    a = np.array(img)
    ys, xs = np.where(a[..., 3] > 24)
    return xs, ys


def _on_white(img: Image.Image) -> Image.Image:
    """Composite an RGBA tile onto white -> RGB (what is actually shown)."""
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")


def pixdiff(a: Image.Image, b: Image.Image, thresh: int = 18) -> int:
    """Count visibly-changed pixels AS DISPLAYED (composited on white).

    Comparing raw RGBA would count antialiased alpha-only differences that are
    invisible on the page -- e.g. mirroring a symmetric icon -- and wrongly accept
    a non-difference. Compositing on white first measures the real visible change.
    """
    aa = np.asarray(_on_white(a)).astype(int)
    bb = np.asarray(_on_white(b)).astype(int)
    return int((np.abs(aa - bb).sum(2) > thresh).sum())


class IconLibrary:
    def __init__(self, manifest_path: Path = MANIFEST, default_split: str = "train"):
        self.default_split = default_split   # pick() uses this unless overridden
        manifest_path = Path(manifest_path).resolve()
        m = json.loads(manifest_path.read_text())
        configured_root = Path(m["icons_root"])
        self.root = (
            configured_root
            if configured_root.is_absolute()
            else ((_ROOT / 'assets/icons') if manifest_path == MANIFEST else manifest_path.parent / configured_root).resolve()
        )
        if not self.root.exists():               # icons_root not found: use the bundled tree
            self.root = (_ROOT / 'assets/icons')
        self.icons = m["icons"]
        # pools keyed by (style, split) and (None-style, split); split defaults
        # to "train" so generators never touch held-out test icons unless asked.
        self._pool = {}
        for i, e in enumerate(self.icons):
            sp = e.get("split", "train")
            self._pool.setdefault((e["style"], sp), []).append(i)
            self._pool.setdefault((None, sp), []).append(i)

    # ---------------------------------------------------------------- selection
    def pick(self, rng, style: str | None = None, min_elem: int = 1,
             split: str | None = None):
        """Pick a reproducible icon. `split` defaults to the library's
        `default_split` ('train'), so training data never uses the held-out
        'test' icons (set `default_split='test'` or pass split='test' for eval)."""
        split = split if split is not None else self.default_split
        pool = list(self._pool.get((style, split), []))
        if min_elem > 1:
            pool = [i for i in pool if self.icons[i]["n_elem"] >= min_elem]
        if not pool:
            raise ValueError(f"no icons for style={style!r} split={split!r}")
        return self.icons[pool[rng.randrange(len(pool))]]

    def noun(self, entry) -> str:
        """A display noun for the question text ('icons' if not a clean word)."""
        nm = entry["name"].strip()
        if entry["set"] == "openmoji":
            return "icons"
        parts = nm.split()
        if 1 <= len(parts) <= 2 and all(p.isalpha() for p in parts):
            base = nm
            plural = base + ("es" if base[-1] in "sxz" or base.endswith(("ch", "sh"))
                             else "s")
            return plural
        return "icons"

    # ---------------------------------------------------------------- rendering
    @staticmethod
    def _is_bg_path(d: str) -> bool:
        """True for game-icons' full-canvas background rect, e.g. 'M0 0h512v512H0z'."""
        if not d:
            return False
        s = d.strip().lower().replace(" ", "").replace(",", "")
        return s.startswith("m00h")

    def _render_gameicon(self, entry, S: int, color) -> Image.Image:
        """game-icons are a black full-canvas rect + a white icon path on top.

        Tinting every opaque pixel would yield a solid square, so instead drop the
        background rect and paint the remaining icon path(s) in the tint color,
        keeping the path's own fill-rule holes as the silhouette's negative space.
        """
        if not HAVE_CAIROSVG:
            raise RuntimeError("rendering icons needs cairosvg; install the generator's dependencies")
        hexc = "#%02x%02x%02x" % (color or (30, 30, 30))
        try:
            tree = ET.parse(self.root / entry["path"])
        except Exception:                                   # fall back to tint
            png = _svg2png(
                path=self.root / entry["path"], width=S, height=S)
            return _tint(Image.open(io.BytesIO(png)).convert("RGBA"),
                         color or (30, 30, 30))
        root = tree.getroot()
        parent = {c: p for p in root.iter() for c in list(p)}
        drop = []
        for e in root.iter():
            if _ln(e.tag) not in DRAW:
                continue
            if _ln(e.tag) == "path" and self._is_bg_path(e.get("d", "")):
                drop.append(e)
                continue
            if e.get("fill") not in (None, "none"):
                e.set("fill", hexc)
            elif e.get("fill") is None:
                e.set("fill", hexc)
            if e.get("stroke") not in (None, "none"):
                e.set("stroke", hexc)
        for e in drop:
            if e in parent:
                parent[e].remove(e)
        out = io.BytesIO()
        tree.write(out, encoding="utf-8", xml_declaration=False)
        png = _svg2png(svg=out.getvalue(), width=S, height=S)
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def render(self, entry, S: int, color=None) -> Image.Image:
        """Render an icon to an RGBA SxS image (transparent bg).

        Silhouette icons are tinted to `color` (default near-black); color icons
        keep their own palette.
        """
        if entry["set"] == "gameicons":
            return self._render_gameicon(entry, S, color)
        if not HAVE_CAIROSVG:
            raise RuntimeError("rendering icons needs cairosvg; install the generator's dependencies")
        png = _svg2png(path=self.root / entry["path"], width=S, height=S)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        if entry["style"] == "silhouette":
            img = _tint(img, color or (30, 30, 30))
        return img

    def _render_removed(self, entry, S: int, rng, color=None):
        """Re-render the icon with ONE visible sub-element deleted (color icons).

        Returns (img, ok). ok=False if the SVG has no removable visible part.
        """
        path = self.root / entry["path"]
        try:
            tree = ET.parse(path)
        except Exception:
            return None, False
        root = tree.getroot()
        parent = {c: p for p in root.iter() for c in list(p)}
        leaves = []
        for e in root.iter():
            if _ln(e.tag) not in DRAW:
                continue
            if e.get("fill") == "none" and e.get("stroke") in (None, "none"):
                continue
            if e in parent:
                leaves.append(e)
        if len(leaves) < 2:                       # keep at least one part visible
            return None, False
        e = leaves[rng.randrange(len(leaves))]
        parent[e].remove(e)
        out = io.BytesIO()
        tree.write(out, encoding="utf-8", xml_declaration=False)
        png = _svg2png(svg=out.getvalue(), width=S, height=S)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        if entry["style"] == "silhouette":
            img = _tint(img, color or (30, 30, 30))
        return img, True

    # ---------------------------------------------------------------- alterations
    # NOTE: pure pose changes (translation "shift", uniform "scale") are
    # intentionally NOT alterations -- the shape itself is unchanged, so they are
    # not valid "find the different" cues. Kept cues genuinely change the icon's
    # structure (erase/add/remove a part), orientation/chirality (rotate/mirror),
    # or color (recolor/hue_shift).
    def _methods(self, entry):
        # color (OpenMoji) icons are multi-part: deleting a real sub-part is the
        # most reliable cue, so prefer it and skip the synthetic 'add' blob.
        if entry["style"] == "color":
            base = ["remove_element", "erase", "hue_shift", "rotate", "mirror"]
            # remove_element deletes a real SVG sub-element and re-rasterises;
            # the other four are pixel operations on the base image.
            return base if HAVE_CAIROSVG else base[1:]
        return ["erase", "add", "rotate", "mirror", "recolor"]

    @staticmethod
    def _stamp(draw, cx, cy, r, fill, rng):
        """Draw one randomly-shaped brush mark; returns its shape name.

        Diversifies erase/add beyond circles: circle, square, triangle, or a
        short thick line (each at a random orientation)."""
        kind = rng.choice(["circle", "square", "triangle", "line"])
        ang = rng.uniform(0, 2 * math.pi)
        if kind == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif kind == "square":
            pts = [(cx + r * math.cos(ang + k * math.pi / 2 + math.pi / 4),
                    cy + r * math.sin(ang + k * math.pi / 2 + math.pi / 4))
                   for k in range(4)]
            draw.polygon(pts, fill=fill)
        elif kind == "triangle":
            pts = [(cx + r * math.cos(ang + k * 2 * math.pi / 3),
                    cy + r * math.sin(ang + k * 2 * math.pi / 3))
                   for k in range(3)]
            draw.polygon(pts, fill=fill)
        else:  # short thick line / bar
            dx, dy, L = math.cos(ang), math.sin(ang), r * 1.9
            w = max(2, int(r * 0.55))
            draw.line([cx - dx * L, cy - dy * L, cx + dx * L, cy + dy * L],
                      fill=fill, width=w)
        return kind

    def _apply(self, entry, base: Image.Image, method: str, strength: float,
               rng, color):
        """Apply one alteration to a rendered SxS RGBA tile. Returns (img, desc)."""
        S = base.size[0]
        s = strength                                   # 0 subtle .. 1 obvious

        if method == "rotate":
            # floor at ~12 deg so it reads as a real orientation change, not an
            # ambiguous tiny tilt
            deg = (12 + s * 28) * rng.choice([1, -1])
            return base.rotate(deg, resample=Image.BICUBIC,
                               center=(S / 2, S / 2)), f"rotated ~{abs(deg):.0f} deg"
        if method == "mirror":
            return base.transpose(Image.FLIP_LEFT_RIGHT), "horizontally mirrored"
        if method == "erase":
            xs, ys = _fg_pixels(base)
            if len(xs) == 0:
                return base, "no-op"
            i = rng.randrange(len(xs))
            r = max(2, int((0.05 + s * 0.10) * S))
            out = base.copy()
            shp = self._stamp(ImageDraw.Draw(out), xs[i], ys[i], r,
                              (0, 0, 0, 0), rng)
            return out, f"a small {shp}-shaped part is missing"
        if method == "add":
            # attach a protrusion at the icon's OUTER edge so it visibly sticks
            # out (stamping inside the same-colored shape would be invisible)
            a = np.array(base)
            mask = a[..., 3] > 24
            if not mask.any():
                return base, "no-op"
            r = max(2, int((0.05 + s * 0.09) * S))
            ring = ndimage.binary_dilation(mask, iterations=max(2, r // 2)) & ~mask
            ys, xs = np.where(ring)
            if len(xs) == 0:
                return base, "no-op"
            i = rng.randrange(len(xs))
            col = color or (30, 30, 30)
            out = base.copy()
            shp = self._stamp(ImageDraw.Draw(out), int(xs[i]), int(ys[i]), r,
                              col + (255,), rng)
            return out, f"an extra {shp}-shaped part sticks out"
        if method == "recolor":                          # silhouette only
            base_col = color or (30, 30, 30)
            shift = int(20 + s * 90)
            newc = tuple(min(255, max(0, c + rng.choice([1, -1]) * shift))
                         for c in base_col)
            return _tint(base, newc), "drawn in a slightly different color"
        if method == "hue_shift":                         # color only
            a = np.array(base.convert("RGBA"))
            rgb = a[..., :3].astype("uint8")
            hsv = np.array(Image.fromarray(rgb, "RGB").convert("HSV"))
            dh = int(10 + s * 60) * rng.choice([1, -1])
            hsv[..., 0] = (hsv[..., 0].astype(int) + dh) % 256
            rgb2 = np.array(Image.fromarray(hsv, "HSV").convert("RGB"))
            a[..., :3] = rgb2
            return Image.fromarray(a, "RGBA"), "the colors are slightly shifted"
        if method == "remove_element":
            img, ok = self._render_removed(entry, S, rng, color)
            if ok:
                return img, "one part of the icon is missing"
            return base, "no-op"
        return base, "no-op"

    def alter(self, entry, base: Image.Image, rng, strength: float, *,
              color=None, lo: int = 12, hi_frac: float = 0.55):
        """Produce ONE subtle, visible alteration of `base`.

        Tries methods (shuffled) at escalating strength; accepts the first whose
        pixel-diff is in a sane band: >= `lo` changed px (perceptible) and <=
        `hi_frac` of the icon's own footprint (still 'the same icon, one change').
        Falls back to a guaranteed-visible erase. Returns (img, method, desc).
        """
        S = base.size[0]
        footprint = max(1, int((np.asarray(base)[..., 3] > 24).sum()))
        hi = int(hi_frac * footprint)
        methods = self._methods(entry)
        rng.shuffle(methods)
        # for multi-part color icons, try deleting a real sub-part first (the
        # most reliable cue)
        if entry["style"] == "color" and "remove_element" in methods:
            methods.remove("remove_element")
            methods.insert(0, "remove_element")
        for m in methods:
            for boost in (1.0, 1.7, 2.6):
                s_eff = min(1.0, strength * boost)
                img, desc = self._apply(entry, base, m, s_eff, rng, color)
                if desc == "no-op":
                    break
                nd = pixdiff(base, img)
                if lo <= nd <= hi or (m == "remove_element" and nd >= lo):
                    return img, m, desc
        # fallback: a clearly-visible erase at the centroid
        xs, ys = _fg_pixels(base)
        out = base.copy()
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            r = max(3, int(0.10 * S))
            ImageDraw.Draw(out).ellipse([cx - r, cy - r, cx + r, cy + r],
                                        fill=(0, 0, 0, 0))
        return out, "erase", "a small part is missing"
