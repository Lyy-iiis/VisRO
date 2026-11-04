"""Independent verifier for `reconstruction` (polyomino assembly, 4-choice).

Imports nothing from the generator (tasks/planar_transformations/reconstruction/generator, generator/base)
or from the adapter.  Everything it needs is either in the record or decoded
from the PNG:

  1. form      question == the declared text, choices A-D, answer a letter,
               meta.correct_index agrees with the letter; the question carries
               no format example (nothing that could equal or share a component
               with the gold: no quoted text, no "e.g."/"for example", no
               standalone option letter outside the "(A-D)" range)
  2. pixels    the image is decoded from scratch: the separator row, the target
               (TARGET_FILL cell interiors -> lattice), the four panels (outline
               columns), and in each panel one piece per piece colour (cell
               interiors -> lattice).  Every cell interior must be a solid
               square of one common size, every cell must sit on its lattice,
               pieces of one panel must be separated by white space, the four
               panel labels must read A, B, C, D (template match against the
               DejaVu font in assets/), and the cells must be legible
               (panel pitch >= 21 px, target pitch >= 18 px).  Residual check:
               every pixel the decode did not read must be the separator line
               (exactly its declared extent), a panel outline (the outline
               colour only on the panel's own 2 px border band and its four
               corner squares; the band must be unbroken and the four
               panels' outlines identical), a label (every box pixel that is
               not a decoded cell or panel outline must equal the matched
               letter rendered with the same font exactly: LABEL_TOL = 0), or
               a decoded cell's outline ring (exactly the outline colour);
               everything else must be pure background, so
               content drawn anywhere else -- off-palette or in the outline
               or label colours -- cannot hide from the decode
  3. image==meta  the decoded target equals meta.target, panel k colour j equals
               meta.display[k][j] cell for cell, and meta.options[k] is a
               rotation-only re-posing of meta.display[k]
  4. rules     tier geometry read off the decoded target ((W,H,cells) declared
               for the tier, rho, connected, spans the box), n pieces per
               option, piece sizes, connectivity, equal areas, 4 distinct option
               sets, 2n distinct shapes, identical (area, perimeter, bbox)
               signature profiles (xhard/xxhard: identical (signature, outline
               corners, bays) profiles), every piece fits inside the target on
               its own
  5. answer    an Algorithm-X exact-cover solver (MRV column choice; not the
               generator's anchor search) on the DECODED pieces: exactly one
               panel tiles the decoded target under the stated rule (move +
               rotate, no flips), it is the answer; every other panel has zero
               tilings under that rule AND with mirroring allowed.  Every search
               that ends without a tiling ran to exhaustion; a search that hits
               the node budget is a problem, never a pass.
  6. solution  meta.solution is a genuine tiling: disjoint placed pieces whose
               union is the target, piece j a rotation (not a mirror) of the
               answer panel's piece j
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import re

import numpy as np
from scipy import ndimage

CORRECTNESS = ("Decodes the target and the four option panels from the PNG pixels alone (every other "
               "pixel must be background, cell outline, separator, panel outline or label), checks them "
               "cell for cell against meta, and re-solves the exact cover with its own Algorithm-X solver: "
               "the answer's panel tiles the decoded target with pieces moved and rotated, never flipped.")
UNIQUENESS = ("The same solver runs each of the other three panels to exhaustion (node-budgeted; hitting "
              "the budget fails the item) and finds zero tilings under the stated rule and also with "
              "mirroring allowed; all option sets have the target's area, so no subset can tile, and "
              "any tiling of a polyomino by polyominoes is grid-aligned, so integer placements are exhaustive.")

QUESTION = (
    "Which option's coloured pieces, each used exactly once, tile the top shape exactly with no "
    'overlap or leftover cells? Pieces may be moved and turned by 90 degrees but not flipped; '
    'options may be drawn at another scale. Answer with the letter.'
)
CHOICES = ["A", "B", "C", "D"]
OPTION_RANGE = "(A-D)"     # names the four panels; not an answer example


def format_examples(question):
    """Everything in the question that could be read as an example answer:
    quoted text, the word after "e.g." / "for example" / "such as" /
    "example", and any standalone option letter outside the "(A-D)" range
    that names the panels.  The release question carries none ("Answer with
    the letter." is the whole format instruction)."""
    q = question.replace(OPTION_RANGE, " ")
    ex = re.findall(r'"([^"]*)"', q)
    ex += re.findall(r"(?:e\.g\.|for example|such as|example)\s*:?\s*\"?([^\s,;\"]+)", q, flags=re.I)
    ex += re.findall(r"(?<![A-Za-z0-9'-])([A-D])(?![A-Za-z0-9'-])", q)
    return [e.strip().strip(".") for e in ex]


def example_overlap(question, answer):
    """-> examples that equal the gold or share a component with it."""
    return [e for e in format_examples(question)
            if e == answer or answer in re.split(r"[^A-Za-z0-9]+", e)]

# Declared tier table, restated here (== generator/tiers.py).  Not imported.
_RECTS = [(8, 6), (6, 8), (12, 4), (7, 7), (10, 5), (13, 4), (9, 6), (11, 5)]
# decoy_key: what every slot's decoy shares with its true piece, so every panel
# shows the same multiset of it ("sig" = (area, perimeter, sorted bbox);
# "sig+corners+bays" adds the outline-corner count and the number of bays --
# connected empty regions inside the piece's bbox -- on the two rigid tiers).
TIERS = {
    "easy":   dict(shapes={(7, 6, 24), (6, 7, 24), (7, 5, 20), (5, 7, 20)}, rho=0.5714,
                   n_parts=4, min_size=3, max_size=10, decoy_key="sig"),
    "medium": dict(shapes={(9, 8, 48)}, rho=0.667, n_parts=8, min_size=4, max_size=10, decoy_key="sig"),
    "hard":   dict(shapes={(8, 7, 48)}, rho=0.857, n_parts=8, min_size=4, max_size=10, decoy_key="sig"),
    "xhard":  dict(shapes={(w, h, w * h - 1) for w, h in _RECTS}, rho=0.980,
                   n_parts=8, min_size=4, max_size=10, decoy_key="sig+corners+bays"),
    "xxhard": dict(shapes={(w, h, w * h) for w, h in _RECTS}, rho=1.000,
                   n_parts=8, min_size=4, max_size=10, decoy_key="sig+corners+bays"),
}

CANVAS = (840, 532)
TARGET_FILL = (96, 110, 128)
PIECE = [(244, 150, 90), (150, 185, 235), (170, 210, 160), (230, 180, 210),
         (240, 210, 130), (180, 170, 225), (140, 205, 205), (235, 165, 160)]
PANEL_LINE = (214, 214, 224)
SEPARATOR = (220, 220, 228)
OUTLINE = (38, 38, 42)     # cell outline colour (target and pieces)
BACKGROUND = (255, 255, 255)
LABEL_BOX = (24, 40)       # rows x cols below a panel's top outline that hold its letter
LABEL_INK = (28, 28, 32)   # label glyph colour; its antialiased edge blends it with white
LABEL_TOL = 0              # per-channel tolerance between a label box and the letter rendered here (16)
CORNER = 12                # panel outline corner square: radius 10 + line width 2
SEP_MARGIN = 10            # the separator line runs from x = 10 to x = W - 10 on each of its rows
PANEL_PX_MIN = 21          # declared legibility floor for option cells
TARGET_PX_MIN = 18         # smallest target cell the layout can produce (147 px // 8 rows)
NODE_BUDGET = 3_000_000    # per exact-cover search; reaching it is a failure
_FONT = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')


class DecodeError(Exception):
    pass


# --------------------------------------------------------------------------- #
# polyomino helpers (own implementation)
# --------------------------------------------------------------------------- #
def _norm(cells):
    cells = [tuple(int(v) for v in c) for c in cells]
    if not cells:
        return ()
    x0 = min(x for x, _ in cells)
    y0 = min(y for _, y in cells)
    return tuple(sorted((x - x0, y - y0) for x, y in cells))


def _rot(cells):
    return _norm([(-y, x) for x, y in cells])


def _mir(cells):
    return _norm([(-x, y) for x, y in cells])


def _poses(cells, mirror):
    out, c = set(), _norm(cells)
    for b in ([c, _mir(c)] if mirror else [c]):
        for _ in range(4):
            out.add(b)
            b = _rot(b)
    return sorted(out)


def _canon(cells):
    return _poses(cells, False)[0]


def _connected(cells):
    s = set(map(tuple, cells))
    if not s:
        return False
    stack, seen = [next(iter(s))], set()
    while stack:
        x, y = stack.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        stack += [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n in s and n not in seen]
    return len(seen) == len(s)


def _corners(cells):
    """outline vertices of a polyomino (a pinch point counts twice)."""
    s = set(map(tuple, cells))
    n = 0
    for vx, vy in {(x + dx, y + dy) for x, y in s for dx in (0, 1) for dy in (0, 1)}:
        q = ((vx - 1, vy - 1) in s, (vx, vy - 1) in s, (vx - 1, vy) in s, (vx, vy) in s)
        k = sum(q)
        n += 1 if k in (1, 3) else (2 if k == 2 and q[0] == q[3] else 0)
    return n


def _bays(cells):
    """connected empty regions (4-neighbour) inside the piece's bounding box"""
    s = set(_norm(cells))
    w = max(x for x, _ in s) + 1
    h = max(y for _, y in s) + 1
    empty = {(x, y) for x in range(w) for y in range(h)} - s
    n = 0
    while empty:
        n += 1
        stack = [empty.pop()]
        while stack:
            x, y = stack.pop()
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if q in empty:
                    empty.discard(q)
                    stack.append(q)
    return n


def _sig(cells):
    s = set(cells)
    per = sum(1 for (x, y) in s for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n not in s)
    xs = [x for x, _ in s]
    ys = [y for _, y in s]
    return (len(s), per, tuple(sorted((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))))


# --------------------------------------------------------------------------- #
# exact cover: Knuth's Algorithm X on dict-of-sets, MRV column choice
# --------------------------------------------------------------------------- #
def tilings(target, pieces, mirror, limit=1, budget=NODE_BUDGET):
    """-> (solutions found (<= limit), nodes, exhausted).  exhausted=False means
    the budget ran out and nothing may be concluded."""
    T = sorted(set(map(tuple, target)))
    if sum(len(p) for p in pieces) != len(T):
        return 0, 0, True
    cid = {c: i for i, c in enumerate(T)}
    n = len(T)
    rows = []
    for i, p in enumerate(pieces):
        seen = set()
        for pose in _poses(p, mirror):
            ax, ay = pose[0]
            for tx, ty in T:
                dx, dy = tx - ax, ty - ay
                ids = [cid.get((x + dx, y + dy)) for x, y in pose]
                if None in ids:
                    continue
                key = frozenset(ids)
                if key in seen:
                    continue
                seen.add(key)
                rows.append([n + i] + ids)
    X = {j: set() for j in range(n + len(pieces))}
    for r, cols in enumerate(rows):
        for j in cols:
            X[j].add(r)
    st = {"sol": 0, "nodes": 0, "cap": False}

    def select(r):
        cols = []
        for j in rows[r]:
            for i in X[j]:
                for k in rows[i]:
                    if k != j:
                        X[k].discard(i)
            cols.append(X.pop(j))
        return cols

    def deselect(r, cols):
        for j in reversed(rows[r]):
            X[j] = cols.pop()
            for i in X[j]:
                for k in rows[i]:
                    if k != j:
                        X[k].add(i)

    def solve():
        if not X:
            st["sol"] += 1
            return
        c = min(X, key=lambda j: len(X[j]))
        for r in sorted(X[c]):
            st["nodes"] += 1
            if st["nodes"] > budget:
                st["cap"] = True
                return
            cols = select(r)
            solve()
            deselect(r, cols)
            if st["sol"] >= limit or st["cap"]:
                return

    solve()
    return st["sol"], st["nodes"], not st["cap"]


# --------------------------------------------------------------------------- #
# pixel decoder (own implementation)
# --------------------------------------------------------------------------- #
def _near(a, rgb, tol=4):
    return np.abs(a - np.array(rgb, dtype=np.int16)).sum(axis=2) <= tol


def _components(mask):
    lab, n = ndimage.label(mask)        # 4-connectivity
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum_labels(mask, lab, idx)
    cents = ndimage.center_of_mass(mask, lab, idx)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab)):
        out.append(dict(cx=float(cents[k][1]), cy=float(cents[k][0]), area=int(areas[k]),
                        w=sl[1].stop - sl[1].start, h=sl[0].stop - sl[0].start,
                        x0=sl[1].start, y0=sl[0].start))
    return out


def _lattice(comps, what):
    """cell interiors -> (normalised integer cells, pitch).  Every interior must
    be a solid square of one common size and have an orthogonal neighbour at
    exactly one pitch, and every centroid must sit on the lattice."""
    if len(comps) < 2:
        raise DecodeError(f"{what}: {len(comps)} cell(s)")
    sizes = {(c["w"], c["h"]) for c in comps}
    if len(sizes) != 1:
        raise DecodeError(f"{what}: cell interiors of different sizes {sorted(sizes)[:4]}")
    (w, h), = sizes
    if w != h or any(c["area"] != w * h for c in comps):
        raise DecodeError(f"{what}: a cell interior is not a solid square")
    pts = np.array([(c["cx"], c["cy"]) for c in comps])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    pitch = float(np.median(nn))
    if np.abs(nn - pitch).max() > 0.01:
        raise DecodeError(f"{what}: a cell has no neighbour at the cell pitch {pitch:.2f}")
    g = (pts - pts.min(axis=0)) / pitch
    if np.abs(g - np.round(g)).max() > 0.02:
        raise DecodeError(f"{what}: cells off the {pitch:.2f}px lattice")
    cells = [(int(x), int(y)) for x, y in np.round(g)]
    if len(set(cells)) != len(cells):
        raise DecodeError(f"{what}: two cells on one lattice point")
    return _norm(cells), pitch, pts


def _runs(idx, max_gap=3):
    groups = []
    for x in idx:
        if groups and x - groups[-1][-1] <= max_gap:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return groups


_TEMPLATES = None


def _letter_renders():
    """{letter: (RGB render on white, row and column of its dark pixels'
    top-left corner)} -- each label letter drawn here with the DejaVu font,
    at the label size, in LABEL_INK."""
    global _TEMPLATES
    if _TEMPLATES is None:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(_FONT), 18)
        out = {}
        for ch in CHOICES:
            im = Image.new("RGB", (40, 40), (255, 255, 255))
            ImageDraw.Draw(im).text((5, 5), ch, fill=LABEL_INK, font=font)
            g = np.asarray(im, dtype=np.int16)
            ys, xs = np.nonzero(g.mean(axis=2) < 128)
            out[ch] = (g, int(ys.min()), int(xs.min()))
        _TEMPLATES = out
    return _TEMPLATES


def _letter_templates():
    """{letter: boolean mask of its dark pixels, cropped}"""
    return {ch: _crop(g.mean(axis=2) < 128) for ch, (g, _y, _x) in _letter_renders().items()}


def _crop(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask[:0, :0]
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _iou(a, b):
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float((a & b).sum()) / max(1, float((a | b).sum()))


def decode(image):
    """PNG -> dict(target, pitch_t, panels[k][j] (colour j), pitch_p, labels)."""
    a = np.asarray(image.convert("RGB"), dtype=np.int16)
    if (a.shape[1], a.shape[0]) != CANVAS:
        raise DecodeError(f"canvas {a.shape[1]}x{a.shape[0]} != {CANVAS}")
    Hh, Ww = a.shape[:2]
    sep_rows = np.nonzero(_near(a, SEPARATOR, 0).sum(axis=1) > 0.9 * Ww)[0]
    if len(sep_rows) == 0:
        raise DecodeError("no separator row between the target and the options")
    s0, s1 = int(sep_rows.min()), int(sep_rows.max())
    if s1 - s0 > 3:
        raise DecodeError("separator row ambiguous")

    tmask = _near(a, TARGET_FILL)
    if tmask[s0:].any():
        raise DecodeError("target colour below the separator")
    pmasks = [_near(a, c) for c in PIECE]
    if any(m[:s1 + 1].any() for m in pmasks):
        raise DecodeError("piece colour above the separator")
    tcomps = _components(tmask[:s0])
    target, pitch_t, _ = _lattice(tcomps, "target")
    cell_boxes = [(c, pitch_t, 0) for c in tcomps]

    # panels: vertical runs of the light-grey outline colour below the separator
    zone = a[s1 + 1:]
    lmask = _near(zone, PANEL_LINE, 0)
    cols = np.nonzero(lmask.sum(axis=0) > 0.5 * zone.shape[0])[0]
    edges = [(g[0], g[-1]) for g in _runs(cols)]
    if len(edges) != 8:
        raise DecodeError(f"expected 8 panel edges, found {len(edges)}")
    spans = [(edges[2 * k][1] + 1, edges[2 * k + 1][0]) for k in range(4)]   # interior x range
    widths = [b - a_ for a_, b in spans]
    if max(widths) - min(widths) > 2 or min(widths) < 100:
        raise DecodeError(f"panel widths {widths}")
    covered = np.zeros(Ww, bool)
    for x0, x1 in spans:
        covered[x0:x1] = True
    if any(m[:, ~covered].any() for m in pmasks):
        raise DecodeError("piece colour outside the four panels")

    panels, pitches, tops, boxes = [], [], [], []
    for k, (x0, x1) in enumerate(spans):
        rows = np.nonzero(lmask[:, x0:x1].sum(axis=1) > 0.5 * (x1 - x0))[0]
        if len(rows) < 2:
            raise DecodeError(f"panel {CHOICES[k]}: no top/bottom outline")
        tops.append(int(rows.min()) + s1 + 1)
        top_run = _runs(rows, max_gap=1)[0]
        # outer box of the outline and its line width (columns, rows)
        boxes.append(dict(L=int(edges[2 * k][0]), R=int(edges[2 * k + 1][1]),
                          T=tops[-1], B=int(rows.max()) + s1 + 1,
                          bw=int(edges[2 * k][1] - edges[2 * k][0] + 1), bh=len(top_run)))
        pieces, ppts = [], []
        for j, m in enumerate(pmasks):
            comps = _components(m[:, x0:x1])
            if not comps:
                pieces.append(None)
                continue
            cells, pitch, pts = _lattice(comps, f"panel {CHOICES[k]} colour {j}")
            cell_boxes += [(c, pitch, x0) for c in comps]
            if not _connected(cells):
                raise DecodeError(f"panel {CHOICES[k]} colour {j}: cells do not form one piece")
            pieces.append(cells)
            ppts.append(pts)
            pitches.append(pitch)
        # drop trailing unused colours; a gap in the colour sequence is a defect
        while pieces and pieces[-1] is None:
            pieces.pop()
        if any(p is None for p in pieces):
            raise DecodeError(f"panel {CHOICES[k]}: piece colours not used in order")
        # different pieces never touch: >= pitch + 3 px between their cell centres
        for i in range(len(ppts)):
            for j in range(i + 1, len(ppts)):
                d = np.sqrt(((ppts[i][:, None, :] - ppts[j][None, :, :]) ** 2).sum(-1)).min()
                if d < pitches[0] + 3:
                    raise DecodeError(f"panel {CHOICES[k]}: pieces {i} and {j} touch")
        panels.append(pieces)
    if max(pitches) - min(pitches) > 0.01:
        raise DecodeError("option panels drawn at different cell pitches")

    # labels: the glyph in each panel's top-left corner must read A, B, C, D
    temps = _letter_templates()
    labels, glyphs = [], []
    renders = _letter_renders()
    lh, lw = LABEL_BOX
    for k, (x0, x1) in enumerate(spans):
        y0 = tops[k]
        # rows y0..y0+23, cols x0..x0+39 should hold only the label.  The
        # renderer's packer can occasionally put a piece inside that box; the
        # glyph then fails the IoU test, the label reads "?" and the item is
        # rejected, the safe direction.
        dark = a[y0:y0 + lh, x0:x0 + lw].mean(axis=2) < 128
        crop = _crop(dark)
        best = max(CHOICES, key=lambda ch: _iou(crop, temps[ch]))
        labels.append(best if _iou(crop, temps[best]) >= 0.8 else "?")
        # what the box must look like: the matched letter rendered here, its
        # dark pixels' top-left corner where the image's dark pixels start
        expect = np.full((lh, lw, 3), 255, np.int16)
        ys, xs = np.nonzero(dark)
        if len(xs):
            g, ty, tx = renders[best]
            oy, ox = int(ys.min()) - ty, int(xs.min()) - tx      # render origin, box coordinates
            ry0, rx0 = max(0, -oy), max(0, -ox)
            ry1, rx1 = min(g.shape[0], lh - oy), min(g.shape[1], lw - ox)
            if ry1 > ry0 and rx1 > rx0:
                expect[oy + ry0:oy + ry1, ox + rx0:ox + rx1] = g[ry0:ry1, rx0:rx1]
        glyphs.append(expect)
    _residual_check(a, s0, s1, spans, tops, cell_boxes, boxes, glyphs, labels)
    return dict(target=target, pitch_t=pitch_t, panels=panels, pitch_p=pitches[0], labels=labels)


def _residual_check(a, s0, s1, spans, tops, cell_boxes, boxes, glyphs, labels):
    """Every pixel the decoder did not read must be accounted for:
      * a decoded cell's interior, or that cell's outline ring (which must be
        exactly OUTLINE);
      * the separator: on each separator row, exactly columns SEP_MARGIN ..
        W - SEP_MARGIN, which must all be the separator colour (or covered by
        a decoded cell), so the line can be neither broken nor extended;
      * a panel outline: the outline colour, and only inside that panel's
        border band (bw columns / bh rows wide along its outer box) or its
        four CORNER x CORNER corner squares.  The band outside the corner
        squares must be unbroken, and the four panels' outline-colour masks
        must be identical (the renderer draws the same rounded rectangle
        four times), so no panel can be marked by adding, removing or
        reshaping outline pixels.  Pixels covered by a decoded cell are left
        out of both tests: the renderer's packer can draw a piece over its
        panel's bottom outline (e.g. hard train s20261011 #30);
      * a label: every pixel of the label box that is not a decoded cell
        (interior or ring) or a panel-outline pixel must equal the matched
        letter rendered here with the DejaVu font, placed where the glyph
        was read (white elsewhere in the box), exactly: LABEL_TOL = 0, so a
        pixel one level off in one channel fails.  So no mark in the box, however
        faint, and no thickened, haloed, thinned or recoloured letter
        passes.  The pixels left out are held by the other rules: a cell
        ring must be exactly OUTLINE, a cell interior is read within 4
        levels summed over the channels (the palette tolerance of _near,
        as everywhere), an outline pixel must pass the outline tests.  A
        box whose label did not read as a letter ("?") is left to the label
        check, which fails the item anyway (a piece drawn into the box).
    Anything else must be pure background, so content outside the palette
    (an extra off-palette cell, a caption, a stray mark) and content drawn in
    the outline or label colours cannot hide from the decode."""
    Hh, Ww = a.shape[:2]
    interior = np.zeros((Hh, Ww), bool)
    ring = np.zeros((Hh, Ww), bool)
    for c, pitch, dx in cell_boxes:
        w = c["w"]
        ow2 = int(round(pitch)) + 1 - w        # outline drawn ow px inside each cell edge
        if ow2 < 2 or ow2 % 2:
            raise DecodeError(f"cell outline width inconsistent (pitch {pitch:.2f}, interior {w})")
        ow = ow2 // 2
        x0, y0 = c["x0"] + dx, c["y0"]
        interior[y0:y0 + w, x0:x0 + w] = True
        ring[max(0, y0 - ow):y0 + w + ow, max(0, x0 - ow):x0 + w + ow] = True
    ring &= ~interior
    off = ring & ~_near(a, OUTLINE, 0)
    if off.any():
        ys, xs = np.nonzero(off)
        raise DecodeError(f"{int(off.sum())} cell-outline pixels not in the outline colour "
                          f"(x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()})")
    cells = interior | ring

    explained = np.zeros((Hh, Ww), bool)
    sep = _near(a, SEPARATOR, 0)
    sx0, sx1 = SEP_MARGIN, Ww - SEP_MARGIN
    for y in range(s0, s1 + 1):
        if not (sep[y, sx0:sx1 + 1] | cells[y, sx0:sx1 + 1]).all():
            raise DecodeError(f"separator row {y} is broken between x {sx0} and {sx1}")
        explained[y, sx0:sx1 + 1] |= sep[y, sx0:sx1 + 1]
    line = _near(a, PANEL_LINE, 0)
    line[:s1 + 1] = False
    sizes = {(b["R"] - b["L"], b["B"] - b["T"], b["bw"], b["bh"]) for b in boxes}
    if len(sizes) != 1:
        raise DecodeError(f"panel outlines differ in size or line width {sorted(sizes)}")
    (pw, ph, bw, bh), = sizes
    if not (1 <= bw <= 4 and 1 <= bh <= 4) or min(pw, ph) < 4 * CORNER:
        raise DecodeError(f"panel outline geometry {pw}x{ph}, line {bw}x{bh}")
    yy, xx = np.mgrid[0:ph + 1, 0:pw + 1]
    band = (xx < bw) | (pw - xx < bw) | (yy < bh) | (ph - yy < bh)
    corner = ((xx < CORNER) | (pw - xx < CORNER)) & ((yy < CORNER) | (ph - yy < CORNER))
    wins = []
    for k, b in enumerate(boxes):
        L, T = b["L"], b["T"]
        win = line[T:T + ph + 1, L:L + pw + 1]
        cov = cells[T:T + ph + 1, L:L + pw + 1]
        if win.shape != band.shape:
            raise DecodeError(f"panel {CHOICES[k]}: outline box leaves the image")
        if not (win | cov)[band & ~corner].all():
            raise DecodeError(f"panel {CHOICES[k]}: outline band broken")
        explained[T:T + ph + 1, L:L + pw + 1] |= win & (band | corner)
        wins.append((win, cov))
    ref, rcov = wins[0]
    for k, (win, cov) in enumerate(wins[1:], 1):
        diff = (win ^ ref) & ~(cov | rcov)
        if diff.any():
            raise DecodeError(f"panel {CHOICES[k]}: outline differs from panel A's ({int(diff.sum())} pixels)")
    lh, lw = LABEL_BOX
    for k, ((x0, _x1), top, expect, lab) in enumerate(zip(spans, tops, glyphs, labels)):
        if lab != "?":
            got = a[top:top + lh, x0:x0 + lw]
            free = ~(cells | explained)[top:top + lh, x0:x0 + lw]
            bad = free & (np.abs(got - expect[:got.shape[0], :got.shape[1]]) > LABEL_TOL).any(axis=2)
            if bad.any():
                raise DecodeError(f"panel {CHOICES[k]}: label box differs from the letter "
                                  f"{lab} rendered here ({int(bad.sum())} pixels)")
        explained[top:top + lh, x0:x0 + lw] = True
    stray = ~(explained | cells) & ~_near(a, BACKGROUND, 0)
    if stray.any():
        ys, xs = np.nonzero(stray)
        raise DecodeError(f"{int(stray.sum())} non-background pixels the decode does not explain "
                          f"(x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()})")


# --------------------------------------------------------------------------- #
# the verifier
# --------------------------------------------------------------------------- #
def verify(record, image):
    P = []
    if record.get("question") != QUESTION:
        P.append("question text differs from the declared form")
    q = record.get("question") or ""
    if format_examples(q):
        P.append(f"question carries a format example {format_examples(q)} (release questions carry none)")
    if example_overlap(q, record.get("answer") or ""):
        P.append("a format example in the question equals or shares a component with the gold")
    if record.get("answer_type") != "choice" or record.get("choices") != CHOICES:
        P.append("answer_type/choices are not choice A-D")
    ans = record.get("answer")
    if ans not in CHOICES:
        return P + [f"answer {ans!r} is not one of A-D"]
    ai = CHOICES.index(ans)
    spec = TIERS.get(record.get("tier"))
    if spec is None:
        return P + [f"unknown tier {record.get('tier')!r}"]
    m = record.get("meta") or {}
    try:
        m_target = _norm(m["target"])
        m_display = [[_norm(p) for p in panel] for panel in m["display"]]
        m_options = [[_norm(p) for p in o] for o in m["options"]]
        m_solution = [[tuple(c) for c in p] for p in m["solution"]]
        if m["correct_index"] != ai:
            P.append(f"meta.correct_index {m['correct_index']} disagrees with answer {ans}")
        if len(m_display) != 4 or len(m_options) != 4:
            P.append("meta must hold exactly 4 panels / option sets")
            return P
    except (KeyError, TypeError, ValueError) as exc:
        return P + [f"meta malformed: {exc!r}"]

    # ---- 2. decode the picture ------------------------------------------- #
    try:
        dec = decode(image)
    except DecodeError as exc:
        return P + [f"image decode: {exc}"]
    target, panels = dec["target"], dec["panels"]
    if dec["labels"] != CHOICES:
        P.append(f"panel labels read {dec['labels']}, expected A-D left to right")
    if dec["pitch_p"] < PANEL_PX_MIN - 0.01:
        P.append(f"option cells {dec['pitch_p']:.1f}px < {PANEL_PX_MIN}px")
    if dec["pitch_t"] < TARGET_PX_MIN - 0.01:
        P.append(f"target cells {dec['pitch_t']:.1f}px < {TARGET_PX_MIN}px")

    # ---- 3. the picture shows exactly what meta says ------------------------ #
    if target != m_target:
        P.append("decoded target differs from meta.target")
    for k in range(4):
        if panels[k] != m_display[k]:
            P.append(f"decoded panel {CHOICES[k]} differs from meta.display (cell for cell, colour order)")
        if sorted(map(_canon, m_display[k])) != sorted(map(_canon, m_options[k])):
            P.append(f"meta.display[{k}] is not a rotation-only re-posing of meta.options[{k}]")

    # ---- 4. tier rules, read off the decoded picture ------------------------ #
    xs = [x for x, _ in target]
    ys = [y for _, y in target]
    w, h, n = max(xs) + 1, max(ys) + 1, len(target)
    if (w, h, n) not in spec["shapes"]:
        P.append(f"target {w}x{h}/{n} cells is not a declared shape of tier {record['tier']}")
    if abs(n / (w * h) - spec["rho"]) >= 0.002:
        P.append(f"target rho {n / (w * h):.4f} != {spec['rho']}")
    if not _connected(target):
        P.append("target is not connected")
    if (m.get("tier"), m.get("rect"), m.get("cells"), m.get("n_parts")) != \
            (record["tier"], [w, h], n, spec["n_parts"]):
        P.append("meta tier/rect/cells/n_parts disagree with the record and the decoded target")
    for k, pieces in enumerate(panels):
        L = CHOICES[k]
        if len(pieces) != spec["n_parts"]:
            P.append(f"panel {L} has {len(pieces)} pieces, tier needs {spec['n_parts']}")
        if sum(len(p) for p in pieces) != n:
            P.append(f"panel {L} has {sum(len(p) for p in pieces)} cells, target has {n}")
        if any(not (spec["min_size"] <= len(p) <= spec["max_size"]) for p in pieces):
            P.append(f"panel {L} has a piece outside {spec['min_size']}-{spec['max_size']} cells")
    sets = [tuple(sorted(map(_canon, pieces))) for pieces in panels]
    if len(set(sets)) != 4:
        P.append("two panels show the same piece set")
    if len({s for ps in sets for s in ps}) != 2 * spec["n_parts"]:
        P.append("the four panels do not use exactly 2n distinct piece shapes")
    sigs = [sorted(_sig(p) for p in pieces) for pieces in panels]
    if any(s != sigs[0] for s in sigs):
        P.append("panels differ in their (area, perimeter, bbox) signature profile")
    if spec["decoy_key"] == "sig+corners+bays":
        prof = [sorted((_sig(p), _corners(p)) for p in pieces) for pieces in panels]
        if any(s != prof[0] for s in prof):
            P.append("panels differ in their (signature, outline corners) profile")
        prof = [sorted((_sig(p), _corners(p), _bays(p)) for p in pieces) for pieces in panels]
        if any(s != prof[0] for s in prof):
            P.append("panels differ in their (signature, outline corners, bays) profile")
    tset = set(target)
    for k, pieces in enumerate(panels):
        for p in pieces:
            if not any(all((x + tx - pose[0][0], y + ty - pose[0][1]) in tset for x, y in pose)
                       for pose in _poses(p, False) for tx, ty in target):
                P.append(f"panel {CHOICES[k]} has a piece that fits nowhere inside the target "
                         f"(every decoy is carved from the target)")
                break
    if P:
        return P

    # ---- 5. re-solve the decoded puzzle ------------------------------------ #
    tiles = []
    for k, pieces in enumerate(panels):
        sol, nodes, done = tilings(target, pieces, mirror=False)
        if not done:
            return P + [f"panel {CHOICES[k]}: rotation-only search hit the {NODE_BUDGET}-node budget"]
        if sol:
            tiles.append(CHOICES[k])
    if tiles != [ans]:
        return P + [f"panels that tile under the stated rule: {tiles}; answer is {ans}"]
    for k, pieces in enumerate(panels):
        if k == ai:
            continue
        sol, nodes, done = tilings(target, pieces, mirror=True)
        if not done:
            return P + [f"panel {CHOICES[k]}: mirror-allowed search hit the {NODE_BUDGET}-node budget"]
        if sol:
            P.append(f"panel {CHOICES[k]} tiles the target if pieces may be flipped")

    # ---- 6. the stored solution is a real tiling ---------------------------- #
    want = panels[ai]
    if len(m_solution) != len(want):
        P.append("meta.solution has the wrong number of pieces")
    else:
        cover = [c for p in m_solution for c in p]
        if len(cover) != len(set(cover)) or set(cover) != set(m_target):
            P.append("meta.solution pieces do not tile meta.target exactly")
        for j, (placed, shown) in enumerate(zip(m_solution, want)):
            if _norm(placed) not in _poses(shown, False):
                P.append(f"meta.solution piece {j} is not a rotation of panel {ans} piece {j}")
                break
    return P
