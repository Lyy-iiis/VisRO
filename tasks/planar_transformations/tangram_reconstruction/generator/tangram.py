"""Release generator for tangram_reconstruction: all five tiers in one standalone module.

  easy, medium  sample_low(): the constructive build, then render()
  hard..xxhard  sample_high(): the sampling loop, then render()

The polytan helpers (sample_polytan, split_target, normalize, rot90, draw_tris, bbox_wh,
connected) are imported from generator/polytan.py.  draw_lattice and label (restated from
shared/puzzles/drawing.py) are a handful of lines each, so this module does not import the
other puzzle tasks.

Given the same random.Random stream, sample() always consumes the stream the same way and
render() draws the same pixels.  Rules:

  * question(): the contract states that exactly one assembly is valid, that k is the
    smallest quarter-turn count giving the placed piece, that piece triangles land on
    target triangles (whole-unit moves), that drawings may use different pixel scales,
    and (easy) that pieces may not be rotated.  The format is shown with letters only
    ([[x,y,k],[x,y,k],...]), so it can never equal or share a value with a real answer.
    The rules are stated in under 256 tokens for LLaVA and PaliGemma (at most 231 with BOS).
  * count_assemblies(): an exact-cover counter used by the release adapter to reject
    every candidate with a second valid answer under the stated rules.
  * diamond_only(): pieces made only of whole diamonds (two triangles sharing a square
    side) are the only shapes whose turns k and k+2 can coincide physically while
    their triangle sets differ, and the only shapes that fit half a diagonal off the
    lattice; the release adapter rejects them.
  * fonts come from the repository's assets/fonts (sha-pinned; no silent system fallback).
  * meta carries the render layout so an independent checker can decode the image.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visro.paths import ROOT as _ROOT

from . import polytan

FONT_PATH = (_ROOT / 'assets/fonts/DejaVuSans.ttf')
FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"

SIZE = 448
ALL_TURNS = (0, 1, 2, 3)

# Tier parameters.  parts = connected parts the silhouette is split into; hint = the
# first (shuffled) part stays on the board as the blue fixed piece; span = bbox limit.
# easy/medium: the constructive builder below; hard..xxhard: the sampling loop.
CONFIG = {
    "easy":   dict(parts=3, area=18, min_size=4, span=6,  hint=True,  turns=[0],          renderer="low"),
    "medium": dict(parts=4, area=24, min_size=4, span=8,  hint=True,  turns=[0, 1, 2, 3], renderer="low"),
    "hard":   dict(parts=4, area=28, min_size=4, span=11, hint=True,  turns=[0, 1, 2, 3], renderer="high"),
    "xhard":  dict(parts=6, area=44, min_size=4, span=11, hint=False, turns=[0, 1, 2, 3], renderer="high"),
    "xxhard": dict(parts=8, area=64, min_size=4, span=11, hint=False, turns=[0, 1, 2, 3], renderer="high"),
}

TARGET = (200, 203, 210)
FIXED = (119, 171, 224)
PIECE = (219, 148, 93)
LABEL = (25, 30, 40)
LATTICE = (175, 181, 188)
DIAGONAL = (206, 209, 214)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def font(size: int):
    if not FONT_PATH.is_file() or _sha(FONT_PATH) != FONT_SHA256:
        raise RuntimeError(f"pinned font missing or changed: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size)


# --------------------------------------------------------------------------- #
# Geometry (public rule, as the question states it)
# --------------------------------------------------------------------------- #
def rotated(cells, k):
    """geometry.rotated: k clockwise quarter turns, normalised to the bbox origin."""
    cells = set(map(tuple, cells))
    for _ in range(k):
        cells = polytan.rot90(cells)
    return polytan.normalize(cells)


def across(c):
    """The triangle on the other side of c's square side (N of (x,y) <-> S of (x,y-1), ...)."""
    x, y, t = c
    return [(x, y - 1, 2), (x + 1, y, 3), (x, y + 1, 0), (x - 1, y, 1)][t]


def diamond_only(cells):
    """True if every triangle's across-the-side neighbour is in the piece too.

    Such a piece is a union of 45-degree diamonds.  Only these shapes survive a half-diagonal
    shift, so only for them can turns k and k+2 (or k and a turn about an edge midpoint) give
    the same physical shape with different triangle sets, or fit the target half a diagonal
    off the lattice.  For every other piece "same placed shape" == "same triangles".
    """
    s = set(map(tuple, cells))
    return bool(s) and all(across(c) in s for c in s)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def piece_scale(pieces):
    """One shared cell size for every piece panel."""
    slot = SIZE / len(pieces)
    limits = [52]
    for p in pieces:
        pw, ph = polytan.bbox_wh(p)
        limits += [int((slot - 30) // pw), 156 // ph]
    return min(limits)


def _orient(parts, rng, turn_of):
    shown, poses = [], []
    for p in parts:
        orig = polytan.normalize(p)
        display = rotated(orig, turn_of())
        shown.append(display)
        undo = next(t for t in range(4) if rotated(display, t) == orig)
        poses.append([min(x for x, _, _ in p), min(y for _, y, _ in p), undo])
    return shown, poses


def sample_low(tier, rng):
    """The constructive build of easy and medium (sampling part)."""
    cfg = CONFIG[tier]
    for attempt in range(4000):
        target = polytan.sample_polytan(rng, cfg["area"])
        if max(polytan.bbox_wh(target)) > cfg["span"]:
            continue
        parts = polytan.split_target(rng, target, cfg["parts"], min_size=cfg["min_size"])
        if not parts:
            continue
        parts = list(parts)
        rng.shuffle(parts)
        fixed = parts.pop(0) if cfg["hint"] else frozenset()
        shown, poses = _orient(parts, rng, lambda: rng.choice(cfg["turns"]))
        if piece_scale([sorted(p) for p in shown]) < 12:
            continue
        return target, fixed, shown, poses, attempt + 1
    raise RuntimeError("Could not construct a low-tier constructive tangram")


def sample_high(tier, rng):
    """The sampling loop of hard, xhard and xxhard."""
    cfg = CONFIG[tier]
    for attempt in range(1000):
        target = polytan.sample_polytan(rng, cfg["area"])
        parts = polytan.split_target(rng, target, cfg["parts"], min_size=cfg["min_size"])
        if parts and max(polytan.bbox_wh(target)) <= cfg["span"]:
            break
    else:
        raise RuntimeError("Cannot split constructive target")
    parts = list(parts)
    rng.shuffle(parts)
    fixed = parts.pop(0) if cfg["hint"] else set()
    shown, poses = _orient(parts, rng, lambda: rng.randrange(4))
    return target, fixed, shown, poses, attempt + 1


def sample(tier, rng):
    return (sample_low if CONFIG[tier]["renderer"] == "low" else sample_high)(tier, rng)


# --------------------------------------------------------------------------- #
# Uniqueness: count every valid answer under a set of allowed turns
# --------------------------------------------------------------------------- #
def count_assemblies(target, fixed, pieces, turns, cap=2):
    """Number of distinct valid answers (piece -> covered triangles maps), up to `cap`.

    Anchor-based exact cover: the smallest uncovered triangle (by y, x, t) must be the
    smallest triangle of whichever piece covers it.  Orientations that cover the same
    triangles are merged, so a rotationally symmetric piece counts once per placement
    (the answer uses the smallest such k).
    """
    order = lambda c: (c[1], c[0], c[2])
    free = frozenset(map(tuple, target)) - frozenset(map(tuple, fixed))
    shapes = []
    for p in pieces:
        seen, mine = set(), []
        for k in turns:
            o = rotated(p, k)
            if o in seen:
                continue
            seen.add(o)
            a = min(o, key=order)
            mine.append((a[2], [(x - a[0], y - a[1], t) for x, y, t in o]))
        shapes.append(mine)
    if sum(len(p) for p in pieces) != len(free):
        return 0
    found = [0]
    used = [False] * len(pieces)

    def solve(rem):
        if found[0] >= cap:
            return
        if not rem:
            found[0] += 1
            return
        ax, ay, at = min(rem, key=order)
        for i, mine in enumerate(shapes):
            if used[i]:
                continue
            for t0, rel in mine:
                if t0 != at:
                    continue
                cells = frozenset((x + ax, y + ay, t) for x, y, t in rel)
                if cells <= rem:
                    used[i] = True
                    solve(rem - cells)
                    used[i] = False

    solve(free)
    return found[0]


# --------------------------------------------------------------------------- #
# Question
# --------------------------------------------------------------------------- #


QUESTION_TURNS = (
    'Cover the gray target with every numbered orange piece once, no gaps or overlaps, leaving '
    'blue triangles uncovered; count grid squares (4 triangles each), not pixels. Pieces may turn'
    ' but not flip. Answer a JSON array without spaces, one [x,y,k] per piece in number order: '
    'turn it k clockwise quarter-turns (0-3, smallest k if equal), then put the top-left of its '
    'grid-aligned bounding box (corner triangles included) at grid point (x,y), x right, y down.'
)
QUESTION_FIXED = (
    'Cover the gray target with every numbered orange piece once, no gaps or overlaps, leaving '
    'blue triangles uncovered; count grid squares (4 triangles each), not pixels. Place each '
    'piece exactly as drawn (no turning or flipping). Answer a JSON array without spaces, one '
    '[x,y,0] per piece in number order: put the top-left of its grid-aligned bounding box (corner'
    ' triangles included) at grid point (x,y), x right, y down.'
)


def question(tier):
    """The tier's question.  It depends only on the allowed turns (easy: none), never on the item."""
    if CONFIG[tier]["turns"] == [0]:
        return QUESTION_FIXED
    return QUESTION_TURNS


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def canvas():
    return Image.new("RGB", (SIZE, SIZE), "white")


def label(im, text, xy, size, labels=None):
    """drawing.label: refuse text outside the canvas."""
    d = ImageDraw.Draw(im)
    f = font(size)
    box = d.textbbox(xy, text, font=f)
    if box[0] < 0 or box[1] < 0 or box[2] > im.width or box[3] > im.height:
        raise RuntimeError(f"Text exceeds image bounds: {text!r} {box}")
    d.text(xy, text, fill=LABEL, font=f)
    if labels is not None:
        labels.append(dict(text=str(text), x=xy[0], y=xy[1], size=size))


def draw_lattice(im, x, y, w, h, cell):
    """The square lattice plus both diagonals of every square."""
    d = ImageDraw.Draw(im)
    for c in range(w + 1):
        d.line((x + c * cell, y, x + c * cell, y + h * cell), fill=LATTICE, width=1)
    for r in range(h + 1):
        d.line((x, y + r * cell, x + w * cell, y + r * cell), fill=LATTICE, width=1)
    for r in range(h):
        for c in range(w):
            d.line((x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell), fill=DIAGONAL, width=1)
            d.line((x + (c + 1) * cell, y + r * cell, x + c * cell, y + (r + 1) * cell), fill=DIAGONAL, width=1)


def render(tier, target, fixed, pieces):
    """Question image plus its layout (cell sizes, origins and label boxes)."""
    low = CONFIG[tier]["renderer"] == "low"
    target = sorted(map(tuple, target))
    fixed = sorted(map(tuple, fixed))
    pieces = [sorted(map(tuple, p)) for p in pieces]
    im = canvas()
    labels = []
    w, h = polytan.bbox_wh(target)
    cell = min(60, 200 // max(w, h)) if low else min(32, 184 // max(w, h))
    ox, oy = (SIZE - w * cell) // 2, 46
    polytan.draw_tris(ImageDraw.Draw(im), target, cell, ox, oy, TARGET)
    if fixed:
        polytan.draw_tris(ImageDraw.Draw(im), fixed, cell, ox, oy, FIXED)
    draw_lattice(im, ox, oy, w, h, cell)
    for c in range(w + 1):
        label(im, str(c), (ox + c * cell - 4, oy - 17), 11, labels)
    for r in range(h + 1):
        label(im, str(r), (ox - 20, oy + r * cell - 7), 11, labels)
    panels = []
    if low:
        pc = piece_scale(pieces)
        if pc < 12:
            raise RuntimeError("Constructive piece cells too small at 448 pixels")
        slot = SIZE / len(pieces)
        for i, p in enumerate(pieces):
            pw, ph = polytan.bbox_wh(p)
            x, y = int(i * slot + (slot - pw * pc) / 2), 282
            label(im, str(i + 1), (x, y - 19), 13, labels)
            polytan.draw_tris(ImageDraw.Draw(im), p, pc, x, y, PIECE)
            draw_lattice(im, x, y, pw, ph, pc)
            panels.append(dict(cell=pc, x=x, y=y, w=pw, h=ph))
    else:
        for i, p in enumerate(pieces):
            r, c = divmod(i, 4)
            x, y = 14 + c * 108, 282 + r * 94
            pw, ph = polytan.bbox_wh(p)
            pc = min(22, 88 // pw, 66 // ph)
            label(im, str(i + 1), (x, y - 18), 13, labels)
            if pc < 10:
                raise RuntimeError("Constructive piece cells too small at 448 pixels")
            polytan.draw_tris(ImageDraw.Draw(im), p, pc, x, y, PIECE)
            draw_lattice(im, x, y, pw, ph, pc)
            panels.append(dict(cell=pc, x=x, y=y, w=pw, h=ph))
    layout = dict(size=SIZE, target=dict(cell=cell, x=ox, y=oy, w=w, h=h), pieces=panels, labels=labels)
    return im, layout


def answer_string(poses):
    return json.dumps([[int(v) for v in p] for p in poses], separators=(",", ":"))
