"""find_the_different: an R x C grid of one icon, exactly one cell altered.

Generator: generator/generate.py; all five tiers are defined by its diff_params.

* Seeding.  `seed` is the framework's derived seed (task, tier, split, base
  seed, index); it is fed to the generator's own derive_rng(seed, index, tier),
  so tiers and splits never share an icon stream.

* Render path.  Every icon is rasterised from the repository SVG with cairosvg
  (gameicons through iconlib's background-stripping path); the adapter refuses
  to run without cairosvg.  The cairosvg and cairo versions are recorded in
  meta.render.

* Alteration parameters.  The exact parameters of the accepted alteration
  (removed SVG leaf, erase/add stamp centre/radius/shape/angle, rotation
  degrees, hue/colour shift) and the strength actually applied are captured by
  recording the RNG draws and the stamp call of iconlib's own code -- nothing
  is re-implemented, so the output is byte-identical to the generator's.  They
  go to meta.alteration_params / meta.applied_strength and a coarse,
  tile-size-independent version to meta.alteration_key.

* Artwork families.  Many library ids share one artwork: 156 groups have
  byte-identical SVGs (e.g. flags of the same design, a gendered emoji whose
  SVG equals the neutral one), 9 more differ only in ids/comments/whitespace
  (mdi/alpha-o = mdi/numeric-0), 64 more groups (152 ids) have different SVGs
  that render pixel-identically at every production size (mdi/alpha-i =
  mdi/roman-numeral-1; the shipped table find_the_different_render_alias.json),
  and OpenMoji skin-tone variants (-1F3FB .. -1F3FF) are the same drawing in
  another skin colour.  Ids are merged over those four relations (union-find
  over the whole manifest) and the family is named by its smallest id:
  meta.icon_family.

* core_key = sha256(task, icon_family, alteration_key).  Grid size, odd
  position, tint, skin tone and tier are deliberately left out: the same
  drawing with the same change is the same discrimination problem, so it may
  appear only once across all five tiers and all splits.

* Icon pool per split: train/eval/val draw from iconlib's train icons, ood from
  the held-out test icons minus the 43 whose artwork is identical to a train
  icon (byte-identical, equal after normalisation, or a pixel-identical render),
  which are not held out.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib
import json
import math
import re

import numpy as np

import shared.iconlib as iconlib
from visro import api
from visro.seeding import digest
from tasks.discrimination_grouping.find_the_different.generator import generate

TASK = "find_the_different"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

_COMMON = ("strength is a base value: the generator multiplies it by 1, 1.25, 1.5, 1.75 or 2 over its retries "
           "and iconlib.alter by 1, 1.7 or 2.6, capped at 1.0 (the value used is meta.applied_strength); "
           "remove_element and mirror ignore it; "
           "icons 60% MDI/game-icons silhouettes in one of 7 tints, 40% OpenMoji colour; "
           "live cairosvg render at 4x, LANCZOS to the tile; odd tile must pass contrast>=64 with an "
           "8-connected component >=48 px and >=12 eroded-core px on the delivered pixels, keep >=48 "
           "visible px, and not be a translated copy; one item per (icon artwork family, alteration) "
           "across tiers and splits, where a family merges ids with identical (normalised) SVG text, pixel-identical "
           "renders or OpenMoji skin tones of one drawing")
# "measured" = medians of a reference run (generate --n 200 --seed 20261011): cells, contrast>=64
# changed px of the odd tile, its largest 8-connected component, and the count of mirror/rotate items.
TIER_DEFINITIONS = {
    "easy": ("448x448 px; rows in {4,5} x cols in {4,5} (tile 88 px for 4x4, else 70 px; 5x5 also occurs in "
             "medium); base strength U(0.70,1.00); no area cap; measured median 20 cells, 158.5 contrast>=64 "
             f"changed px, largest component 150 px, 17/200 mirror/rotate; {_COMMON}"),
    "medium": ("448x448 px; rows in {5,6} x cols in {5,6,7} (tile 70/59/50 px; 5x5 also occurs in easy, 6x6 and "
               "6x7 in hard); base strength U(0.35,0.60); no area cap; measured median 30 cells, 87.5 contrast>=64 "
               "changed px, largest component 78 px, 20/200 mirror/rotate: harder than easy, and NOT easier than "
               f"hard (see hard); {_COMMON}"),
    "hard": ("448x448 px; rows in {6,7,8} x cols in {6,7,8} (tile 59/50/43 px; 6x6 and 6x7 also occur in medium); "
             "base strength U(0.10,0.30); no area cap; measured median 48 cells, 186.5 contrast>=64 changed px, "
             "largest component 110 px, 83/200 mirror/rotate (weak erase/add stamps fail the floor on its small "
             "tiles, so iconlib.alter falls through to mirror/rotate): more cells than medium but larger changes, "
             "so hard is NOT harder than medium, and on the shared 6x7 grid it is easier (257 vs 71.5 changed px); "
             f"{_COMMON}"),
    "xhard": ("588x588 px; 9x9 grid (tile 51 px); base strength U(0.30,0.50); contrast>=64 changed px <= 12% of "
              "tile^2; measured median 81 cells, 73.5 contrast>=64 changed px, largest component 69 px, 10/200 "
              f"mirror/rotate: harder than hard; {_COMMON}"),
    "xxhard": ("588x588 px; 10x10 grid (tile 46 px); base strength U(0.25,0.45); contrast>=64 changed px <= 8% of "
               "tile^2; measured median 100 cells, 63 contrast>=64 changed px, largest component 63 px, 4/200 "
               f"mirror/rotate: modestly harder than xhard; {_COMMON}"),
}

ICON_POOL = {"train": "train", "eval": "train", "val": "train", "ood": "test"}

if not iconlib.HAVE_CAIROSVG:
    raise RuntimeError("find_the_different release items are rendered with cairosvg only; it is not importable")

import cairocffi  # noqa: E402  (a cairosvg dependency; recorded for provenance)
import cairosvg  # noqa: E402

RENDER = {
    "path": "cairosvg-live",
    "cairosvg": cairosvg.__version__,
    "cairo": cairocffi.cairo_version_string(),
    "supersample": generate.SS,
    "downsample": "LANCZOS",
}


class _RngRecorder:
    """Delegates to a random.Random and logs every draw (results unchanged)."""

    def __init__(self, rng):
        self._rng = rng
        self.log = []

    def __getattr__(self, name):
        fn = getattr(self._rng, name)
        if not callable(fn):
            return fn

        def call(*args, **kw):
            out = fn(*args, **kw)
            self.log.append((name, out))
            return out
        return call


class _ReleaseIconLibrary(iconlib.IconLibrary):
    """iconlib.IconLibrary with the render path pinned and alterations recorded.

    The held-out 'test' pools drop icons whose artwork equals a train icon's."""

    def __init__(self, default_split: str = "train"):
        super().__init__(default_split=default_split)
        index = icon_index(self)
        for key in [k for k in self._pool if k[1] == "test"]:
            self._pool[key] = [i for i in self._pool[key] if self.icons[i]["id"] not in index["art_in_train"]]

    def _stamp(self, draw, cx, cy, r, fill, rng):
        self._stamp_at = (float(cx), float(cy), float(r))
        return iconlib.IconLibrary._stamp(draw, cx, cy, r, fill, rng)

    def _apply(self, entry, base, method, strength, rng, color):
        rec = _RngRecorder(rng)
        self._stamp_at = None
        img, desc = super()._apply(entry, base, method, strength, rec, color)
        self._last_apply = (method, float(strength), list(rec.log), self._stamp_at, base.size[0], color)
        return img, desc

    def alter(self, entry, base, rng, strength, **kw):
        self._last_apply = None
        img, method, desc = super().alter(entry, base, rng, strength, **kw)
        if desc == "a small part is missing":       # alter()'s centroid-erase fallback
            self.last_alteration = {"method": "erase_fallback"}
            self.last_applied_strength = None
        else:
            m, s, log, stamp, S, color = self._last_apply
            assert m == method, (m, method)
            self.last_alteration = _params(m, s, log, stamp, S, color)
            self.last_applied_strength = s       # base * generator retry factor * alter boost, <= 1
        return img, method, desc


# ---------------------------------------------------------------- artwork families
_SKIN_TONE = re.compile(r"-1F3F[B-F]")      # OpenMoji Fitzpatrick modifiers in an id


def _svg_normalised(raw: bytes) -> bytes:
    """SVG text without comments, id/data-name attributes and whitespace runs."""
    t = re.sub(rb"<!--.*?-->", b"", raw, flags=re.S)
    t = re.sub(rb'\s(?:id|data-name)="[^"]*"', b"", t)
    return re.sub(rb"\s+", b" ", t).strip()


class IconClasses:
    """Union-find over icon ids; a class is named by its smallest id."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def merge_by(self, ids, key):
        first = {}
        for i in ids:
            k = key(i)
            if k is None:
                continue
            if k in first:
                a, b = self.find(first[k]), self.find(i)
                if a != b:
                    self.parent[max(a, b)] = min(a, b)
            else:
                first[k] = i


_INDEX: dict | None = None

# Render aliases: ids whose SVGs differ (not byte- or normalised-identical) but whose live renders are
# pixel-identical, e.g. mdi/alpha-i = mdi/roman-numeral-1 or openmoji/1F93C-200D-2642-FE0F = an
# openmoji/1F9D1-...-1FAEF-... couple.  Finding them takes a live render of all 16,122 manifest icons
# (about 15 CPU-minutes), so the groups are precomputed by generator/build/render_alias.py and shipped
# next to this file, keyed by a digest of the icon library (ids, sets, styles and SVG bytes).
RENDER_ALIAS_PATH = (_ROOT / 'tasks/discrimination_grouping/find_the_different/find_the_different_render_alias.json')
# Every size an icon is rasterised at in production: generate.SS (4) x every tile the tiers use
# (easy 88/70, medium 70/59/50, hard 59/50/43, xhard 51, xxhard 46 px).
RENDER_SIZES = (172, 184, 200, 204, 236, 280, 352)


def library_digest(icons, sha) -> str:
    """Identity of the icon library: (id, set, style, sha256 of the SVG bytes) of every manifest icon."""
    h = hashlib.sha256()
    for e in sorted(icons, key=lambda e: e["id"]):
        h.update(f"{e['id']}\t{e['set']}\t{e['style']}\t{sha[e['id']]}\n".encode())
    return h.hexdigest()


def svg_classes(lib):
    """(ids, id -> SVG sha256, union-find over byte-identical or normalised-identical SVGs)."""
    ids = [e["id"] for e in lib.icons]
    raw = {e["id"]: (lib.root / e["path"]).read_bytes() for e in lib.icons}
    sha = {i: hashlib.sha256(raw[i]).hexdigest() for i in ids}
    art = IconClasses(ids)
    art.merge_by(ids, sha.get)
    art.merge_by(ids, lambda i: hashlib.sha256(_svg_normalised(raw[i])).hexdigest())
    return ids, sha, art


def _render_aliases(icons, sha) -> list:
    """The shipped render-alias groups; refuses a table computed for another icon library."""
    table = json.loads(RENDER_ALIAS_PATH.read_text())
    if table["library_digest"] != library_digest(icons, sha):
        raise RuntimeError(f"{RENDER_ALIAS_PATH.name} was computed for another icon library; rebuild it with "
                           "python -m tasks.discrimination_grouping.find_the_different.generator.build.render_alias")
    return table["groups"]


def icon_index(lib) -> dict:
    """Artwork classes over the whole manifest (train and test), computed once.

    art     id -> class of ids with byte-identical or normalised-identical SVGs, or pixel-identical
            live renders (the shipped render-alias table, RENDER_ALIAS_PATH)
    family  id -> art classes further merged across OpenMoji skin tones
    art_in_train  test ids whose art class contains a train id (not held out)
    """
    global _INDEX
    if _INDEX is None:
        icons = lib.icons
        ids, sha, art = svg_classes(lib)
        alias = {i: k for k, group in enumerate(_render_aliases(icons, sha)) for i in group}
        art.merge_by(ids, alias.get)
        fam = IconClasses(ids)
        fam.merge_by(ids, art.find)
        fam.merge_by(ids, lambda i: _SKIN_TONE.sub("", i) if i.startswith("openmoji/") else None)
        split = {e["id"]: e.get("split", "train") for e in icons}
        train_art = {art.find(i) for i in ids if split[i] == "train"}
        _INDEX = {
            "art": {i: art.find(i) for i in ids},
            "family": {i: fam.find(i) for i in ids},
            "art_in_train": frozenset(i for i in ids if split[i] == "test" and art.find(i) in train_art),
        }
    return _INDEX


def icon_family(icon_id: str, lib) -> str:
    return icon_index(lib)["family"][icon_id]


def _params(method, s, log, stamp, S, color):
    """Exact parameters of one iconlib._apply call, from its recorded draws."""
    if method == "remove_element":
        (_, k), = [d for d in log if d[0] == "randrange"]
        return {"method": method, "leaf_index": int(k)}
    if method in ("erase", "add"):
        cx, cy, r = stamp
        kind, ang = [d[1] for d in log if d[0] in ("choice", "uniform")][-2:]
        return {"method": method, "shape": kind, "cx": round(cx / S, 4), "cy": round(cy / S, 4),
                "r": round(r / S, 4), "angle": round(ang, 4)}
    if method == "rotate":
        (_, sign), = log
        return {"method": method, "degrees": round((12 + s * 28) * sign, 3)}
    if method == "mirror":
        return {"method": method}
    if method == "recolor":
        shift = int(20 + s * 90)
        signs = [d[1] for d in log]
        base = color or (30, 30, 30)
        return {"method": method, "rgb": [min(255, max(0, c + g * shift)) for c, g in zip(base, signs)]}
    if method == "hue_shift":
        (_, sign), = log
        return {"method": method, "dh": int(10 + s * 60) * sign}
    raise ValueError(f"unknown alteration {method!r}")


def alteration_key(p: dict) -> list:
    """Coarse, tile-size-independent identity of an alteration (for core_key)."""
    m = p["method"]
    if m == "remove_element":
        return [m, p["leaf_index"]]
    if m in ("erase", "add"):                        # stamp centre on a 6x6 grid of the icon box
        return [m, p["shape"], min(5, int(p["cx"] * 6)), min(5, int(p["cy"] * 6))]
    if m == "rotate":
        return [m, int(math.copysign(1, p["degrees"])), round(abs(p["degrees"]) / 10)]
    if m == "recolor":
        return [m, p["rgb"]]
    if m == "hue_shift":
        return [m, int(math.copysign(1, p["dh"]))]
    return [m]                                       # mirror, erase_fallback


_LIBS: dict[str, _ReleaseIconLibrary] = {}


def _library(pool: str) -> _ReleaseIconLibrary:
    if pool not in _LIBS:
        _LIBS[pool] = _ReleaseIconLibrary(default_split=pool)
    return _LIBS[pool]


def core_key(family: str, alt_key: list) -> str:
    return digest([TASK, family, alt_key])


def _tile_sha(img, meta) -> dict:
    """sha256 of the delivered odd tile and of one normal tile (RGB bytes)."""
    a = np.asarray(img.convert("RGB"))
    t, cell, (left, top) = meta["tile_px"], meta["cell_px"], meta["grid_origin"]
    pad = (cell - t) // 2

    def crop(r, c):
        x, y = left + (c - 1) * cell + pad, top + (r - 1) * cell + pad
        return hashlib.sha256(np.ascontiguousarray(a[y:y + t, x:x + t]).tobytes()).hexdigest()
    odd = (meta["odd_row"], meta["odd_col"])
    normal = (1, 1) if odd != (1, 1) else (1, 2)
    return {"odd": crop(*odd), "normal": crop(*normal)}


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in TIER_DEFINITIONS:
        raise ValueError(f"unknown tier {tier!r}")
    pool = ICON_POOL[split]
    lib = _library(pool)
    lib.last_alteration = lib.last_applied_strength = None
    rng = generate.derive_rng(seed, index, tier)
    try:
        img, question, answer, cot, meta = generate.make_problem(lib, rng, None, tier)
    except RuntimeError as exc:                      # no readable odd tile within 160 icons
        raise api.Reject(str(exc)) from None
    params = lib.last_alteration
    if params is None:
        raise api.Reject("alteration parameters were not recorded")
    akey = alteration_key(params)
    family = icon_family(meta["icon_id"], lib)
    image = img.convert("RGB")
    meta = dict(meta)
    meta.update({
        "icon_pool": pool,
        "icon_family": family,
        "render": dict(RENDER),
        "alteration_params": params,
        "alteration_key": akey,
        "applied_strength": lib.last_applied_strength,
        "tile_sha256": _tile_sha(image, meta),
        "cot": cot,
    })
    return api.Item(
        question=question,
        answer=answer,
        image=image,
        meta=meta,
        core_key=core_key(family, akey),
        answer_type="text",
    )
