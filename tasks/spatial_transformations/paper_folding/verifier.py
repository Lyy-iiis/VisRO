"""Independent verifier for paper_folding (staged fold-and-punch, bitstring answer).

Nothing here imports or calls the generator.
The gold answer is recomputed from what a solver is given:

  QUESTION TEXT -> sheet size, fold sequence (axis, crease line, direction), which
                   fold each panel's punches follow, queried cells, digit count.
                   The parsed structure is re-emitted with this module's own copy
                   of the declared template and must reproduce the text exactly,
                   so every convention sentence and the answer format are checked.
  IMAGE PIXELS  -> panels (paper-coloured components), their grid (counted grid
                   lines -> packet width/height in cells), outline shape
                   (rectangle / lower-left / lower-right triangle), the panel
                   titles P1..Pk and the row/column tick numbers (glyph-template
                   OCR with the shipped DejaVuSans.ttf), and the red punch dots
                   (component centroid -> packet cell -> original-sheet cell via
                   the decoded tick numbers).

A forward layer-tracking simulation of the parsed folds with the decoded punches
gives the full hole grid and hence the bitstring, which must equal the record's
answer.  The record's meta (folds, events, truth, queries, tier parameters) and its
core_key are then cross-checked against the text/image-derived structure.

Release policies checked here as well (paper_folding):
  * pixel accounting: every pixel of a panel box is explained by the sheet, a detected
    grid line, the outline, a decoded dot or the white background, and every interior
    grid line is drawn across the packet (no extra palette-coloured marks);
  * structure balance: the 1-cells and 0-cells of the query list have the same multiset
    of text-computable signatures (mate counts per punch stage, on-crease flag), so the
    fold text alone does not reveal which queried cells are holes;
  * no answer-shaped example string in the question.

Both generator routes (meta.template "paper_query" for easy..hard, "folding" for xhard/xxhard)
use one compact question template; this module keeps its own copy (emit_question).
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import collections
import functools
import re

import numpy as np
from scipy import ndimage

from visro.seeding import digest

CORRECTNESS = ("The fold sequence and queried cells are parsed from the question text (which must re-emit "
               "exactly from the declared template); panels, packet grid size and shape, P-titles, row/column "
               "tick numbers (glyph OCR) and red punch dots are decoded from the PNG; a forward layer-tracking "
               "simulation of those folds and punches yields the hole grid and the bitstring, which must equal "
               "the answer, meta.truth, meta.events and meta.folds; core_key must be the digest of the verified "
               "folds and punches; every panel pixel must be explained by sheet, grid line, outline or a decoded "
               "dot; the query list's 1-cells and 0-cells must carry the same text-computable mate/crease "
               "signatures; the text must hold no answer-shaped 0/1 example.")
UNIQUENESS = ("The text fixes every fold (crease on the current packet's midline or the square packet's "
              "diagonal, direction named) and the image fixes every punch cell (each dot within 0.2 cell of a "
              "cell centre, radius < 0.3 cell, on paper, off any diagonal, one per cell, ticks consistent with "
              "the text's packet), so the hole set and hence the requested fixed-length 0/1 string over distinct "
              "listed cells is unique; every punch stage also owns a queried hole no other stage makes.")

# This verifier's own copy of the tier table (checked against every record).
TIERS = {
    "easy": dict(n=8, folds=1, stages=[1], punches=[4], queries=6, diagonal_last=False, template="paper_query"),
    "medium": dict(n=8, folds=2, stages=[1, 2], punches=[2, 2], queries=8, diagonal_last=False, template="paper_query"),
    "hard": dict(n=8, folds=3, stages=[1, 3], punches=[2, 2], queries=10, diagonal_last=True, template="paper_query"),
    "xhard": dict(n=16, folds=4, stages=[2, 3, 4], punches=[1, 1, 2], queries=12, diagonal_last=False, template="folding"),
    "xxhard": dict(n=16, folds=5, stages=[2, 4, 5], punches=[1, 2, 2], queries=16, diagonal_last=True, template="folding"),
}
CANVAS = (448, 448)
FONT_PATH = (_ROOT / 'assets/fonts/DejaVuSans.ttf')

SHEET, GRID, EDGE, RED, WHITE = (218, 229, 242), (155, 174, 195), (45, 60, 78), (200, 42, 51), (255, 255, 255)
TEXT = (25, 30, 40)

# Legibility / determinacy thresholds.
MIN_CELL_PX = 12.0          # smallest packet cell in a punched panel
MIN_DOT_RADIUS_PX = 2.4     # smallest punch dot
MAX_DOT_OFFSET = 0.20       # dot centroid to cell centre, in cells (Chebyshev)
MAX_DOT_RADIUS = 0.30       # dot radius, in cells (keeps the dot inside its cell)
OCR_MAX_DIST = 0.10         # accepted glyph-template distance (0 = identical ink)
OCR_MIN_MARGIN = 0.04       # best vs second-best candidate distance


class Problem(Exception):
    pass


def need(ok, msg):
    if not ok:
        raise Problem(msg)


# --------------------------------------------------------------------------- text

# A fold names its crease (x<k>: vertical grid line k, y<k>: horizontal) and the half that
# MOVES over it ("x8 left" = the half left of x=8 is folded onto the right half, keep="high").
# Every crease bisects the current packet (resolve_folds enforces it), so the moving half
# always lands on the other half and the text does not name the destination.
_MOVING_HALF = {"v": ("right", "left"), "h": ("bottom", "top")}      # [keep == "low", keep == "high"]
_DIAGONAL = {"d": "upper right triangle", "a": "upper left triangle"}
TEMPLATES = ("paper_query", "folding")   # meta.template (generator route); both share the text below 


def _fold_desc(axis: str, line2, keep: str) -> str:
    if axis in _MOVING_HALF:
        return f"{'x' if axis == 'v' else 'y'}{line2 / 2:g} {_MOVING_HALF[axis][keep != 'low']}"
    return _DIAGONAL[axis]


def emit_question(template: str, n: int, folds, stages, queries) -> str:
    """The declared question form.  folds: [(axis, line2, keep)]; stages: fold numbers
    followed by a punch panel, in panel order; queries: 0-based (row, col)."""
    if template not in TEMPLATES:
        raise Problem(f"unknown template {template!r}")
    steps = "; ".join(_fold_desc(*f) + (f", punch P{stages.index(i) + 1}" if i in stages else "")
                      for i, f in enumerate(folds, 1))
    panels = "Panel P1 shows" if len(stages) == 1 else "Each panel shows"
    cells = ",".join(f"({r + 1},{c + 1})" for r, c in queries)
    return (f"Fold the {n}x{n} grid sheet in order without turning it, moving the named half (crease x/y = grid line, "
            f"0 at top left): {steps}; then unfold. {panels} the packet in the original frame with red holes punched "
            f"through all layers (earlier holes omitted). Give one {len(queries)}-digit string: per cell (row,column) "
            f"below in order, 1 if a hole is centred there, else 0: {cells}.")


def example_overlap(question: str, gold: str) -> list[str]:
    """The template carries no example answer.  Any run of three or more 0/1 characters
    in the text would be an answer-shaped example (every gold has >= 6 digits; the
    only 0/1 runs the template produces are the numbers 0, 1, 10 and 11), and the gold
    itself must not occur in the text."""
    out = []
    runs = [m.group(0) for m in re.finditer(r"(?<![0-9])[01]{3,}(?![0-9])", question)]
    if runs:
        out.append(f"question contains answer-shaped 0/1 strings {runs[:3]}")
    if gold and gold in question:
        out.append("question contains the gold answer")
    return out


_KEEP = {("v", "right"): "low", ("v", "left"): "high", ("h", "bottom"): "low", ("h", "top"): "high"}


def parse_question(q: str):
    """Lenient parse; exactness is enforced by re-emission."""
    m = re.match(r"Fold the (\d+)x(\d+) grid sheet ", q)
    need(m and m.group(1) == m.group(2), "question: sheet size not stated as 'Fold the NxN grid sheet'")
    n = int(m.group(1))
    fl = re.search(r"moving the named half \(crease x/y = grid line, 0 at top left\): (.*?); then unfold\.", q)
    need(fl, "question: no fold list")
    folds, stage_of_panel = [], {}
    for i, item in enumerate(fl.group(1).split("; "), 1):
        body, _, panel = item.partition(", punch P")
        am = re.fullmatch(r"([xy])(\d+) (right|left|bottom|top)", body)
        if am:
            axis = "v" if am.group(1) == "x" else "h"
            keep = _KEEP.get((axis, am.group(3)))
            need(keep, f"question: fold {i}: crease {am.group(1)}{am.group(2)} has no {am.group(3)} half")
            folds.append((axis, 2 * int(am.group(2)), keep))
        elif body in _DIAGONAL.values():
            folds.append((next(a for a, t in _DIAGONAL.items() if t == body), None, "lower"))
        else:
            raise Problem(f"question: unparsed fold {i}: {body[:60]!r}")
        if panel:
            need(panel.isdigit(), f"question: fold {i}: bad punch panel {panel[:10]!r}")
            k = int(panel)
            need(k not in stage_of_panel, f"question: panel P{k} used twice")
            stage_of_panel[k] = i
    need(folds, "question: no folds")
    qm = re.search(r"Give one (\d+)-digit string: .*?else 0: (.*)\.$", q)
    need(qm, "question: no query list / digit count")
    queries = [(int(a) - 1, int(b) - 1) for a, b in re.findall(r"\((\d+),(\d+)\)", qm.group(2))]
    need(len(queries) == int(qm.group(1)), "question: digit count != number of queried cells")
    stages = [stage_of_panel[k] for k in sorted(stage_of_panel)]
    need(sorted(stage_of_panel) == list(range(1, len(stages) + 1)), "question: panels not numbered P1..Pk")
    return n, folds, stages, queries


def resolve_folds(n: int, folds):
    """Packet boxes (doubled units) before/after each fold; diagonal crease values.
    Enforces that each axis crease is the current packet's midline and that a
    diagonal fold is applied to a square packet."""
    box = [0, 0, 2 * n, 2 * n]
    out = []
    for i, (axis, line, keep) in enumerate(folds, 1):
        x0, y0, x1, y1 = box
        before = list(box)
        if axis == "v":
            need(x1 - x0 >= 4 and line == (x0 + x1) // 2 and (x0 + x1) % 4 == 0,
                 f"fold {i}: x crease {line / 2:g} is not the packet midline")
            box = [x0, y0, line, y1] if keep == "low" else [line, y0, x1, y1]
        elif axis == "h":
            need(y1 - y0 >= 4 and line == (y0 + y1) // 2 and (y0 + y1) % 4 == 0,
                 f"fold {i}: y crease {line / 2:g} is not the packet midline")
            box = [x0, y0, x1, line] if keep == "low" else [x0, line, x1, y1]
        else:
            need(x1 - x0 == y1 - y0, f"fold {i}: diagonal fold of a non-square packet")
            need(i == len(folds), f"fold {i}: a diagonal fold must be the last fold")
            line = x0 - y0 if axis == "d" else x0 + y1
        out.append(dict(axis=axis, line=line, keep=keep, before=before, after=list(box)))
    return out


def reflect(p, f):
    x, y = p
    a, l, k = f["axis"], f["line"], f["keep"]
    if a == "v":
        return (2 * l - x, y) if ((x > l) if k == "low" else (x < l)) else p
    if a == "h":
        return (x, 2 * l - y) if ((y > l) if k == "low" else (y < l)) else p
    if a == "d":
        return (y + l, x - l) if x - y > l else p
    return (l - y, l - x) if x + y < l else p


def legal_cell(X, Y, f):
    """A punch centre (odd doubled coordinates) strictly inside the packet after fold f."""
    x0, y0, x1, y1 = f["after"]
    if not (x0 < X < x1 and y0 < Y < y1 and X % 2 and Y % 2):
        return False
    if f["axis"] == "d":
        return X - Y < f["line"]
    if f["axis"] == "a":
        return X + Y > f["line"]
    return True


def simulate(n, folds, punches):
    """Forward layer tracking.  punches: {fold number: set of (X, Y)}.
    Returns (holes, per-stage effects), holes as 0-based (row, col)."""
    pos = {(r, c): (2 * c + 1, 2 * r + 1) for r in range(n) for c in range(n)}
    holes, effects = set(), {}
    for i, f in enumerate(folds, 1):
        pos = {k: reflect(p, f) for k, p in pos.items()}
        x0, y0, x1, y1 = f["after"]
        need(all(x0 < x < x1 and y0 < y < y1 for x, y in pos.values()), f"fold {i}: material outside the packet")
        if f["axis"] == "d":
            need(all(x - y <= f["line"] for x, y in pos.values()), "diagonal fold left material on the far side")
        if f["axis"] == "a":
            need(all(x + y >= f["line"] for x, y in pos.values()), "diagonal fold left material on the far side")
        if i in punches:
            stage_new = set()
            for X, Y in sorted(punches[i]):
                hit = {k for k, p in pos.items() if p == (X, Y)}
                need(hit - holes - stage_new, f"stage after fold {i}: punch {(X, Y)} removes no new material")
                stage_new |= hit
            effects[i] = stage_new
            holes |= stage_new
    return holes, effects


def query_signatures(n, folds, stages, queries):
    """What the fold text alone says about each queried cell: for every punch stage the
    number of OTHER queried cells that sit at the same packet position after that fold
    (they are punched together there), and whether the cell lies on the diagonal crease
    when the last fold is a diagonal punch stage (such a cell cannot be punched then)."""
    pos = {q: (2 * q[1] + 1, 2 * q[0] + 1) for q in queries}
    per_stage = {}
    for i, f in enumerate(folds, 1):
        pos = {q: reflect(p, f) for q, p in pos.items()}
        if i in stages:
            per_stage[i] = dict(pos)
    last = folds[-1]
    out = []
    for q in queries:
        sig = tuple(sum(1 for o in queries if o != q and per_stage[s][o] == per_stage[s][q]) for s in stages)
        x, y = pos[q]
        crease = (last["axis"] == "d" and x - y == last["line"]) or (last["axis"] == "a" and x + y == last["line"])
        out.append(sig + (bool(crease and stages[-1] == len(folds)),))
    return out


# --------------------------------------------------------------------------- image

@functools.lru_cache(None)
def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT_PATH), size)


_BOX = (18, 30)   # OCR canvas (rows, cols)
_INK_THR = 0.15


def _crop_ink(alpha):
    ys, xs = np.nonzero(alpha > _INK_THR)
    if not len(ys):
        return None
    return alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _place(crop, dy=0, dx=0):
    """crop placed at (2+dy, 2+dx) on the fixed OCR canvas; None if it does not fit."""
    out = np.zeros(_BOX)
    h, w = crop.shape
    if h + 2 + dy > _BOX[0] or w + 2 + dx > _BOX[1]:
        return None
    out[2 + dy:2 + dy + h, 2 + dx:2 + dx + w] = crop
    return out


@functools.lru_cache(None)
def _templates(text, size):
    """Ink of `text` rendered with the shipped font at 16 sub-pixel phases, placed at the canvas origin."""
    from PIL import Image, ImageDraw
    stack, shapes = [], set()
    for fy in (0, .25, .5, .75):
        for fx in (0, .25, .5, .75):
            im = Image.new("L", (48, 32), 255)
            ImageDraw.Draw(im).text((4 + fx, 4 + fy), text, fill=0, font=_font(size))
            crop = _crop_ink((255 - np.asarray(im, float)) / 255.0)
            shapes.add(crop.shape)
            placed = _place(crop)
            if placed is not None:
                stack.append(placed)
    stack = np.stack(stack)
    return stack, stack.sum(axis=(1, 2)), shapes


def ocr(alpha, candidates, size):
    """Best candidate string for an ink patch.  Returns (text, distance, margin);
    distance = L1(ink - template ink) / (total ink), minimised over sub-pixel phase
    and a +-1 px shift; margin = distance gap to the best other candidate."""
    crop = _crop_ink(alpha)
    if crop is None:
        return None, 1.0, 0.0
    shifted = [o for o in (_place(crop, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) if o is not None]
    if not shifted:  # larger than any label or title
        return None, 1.0, 0.0
    shifted = np.stack(shifted)
    osum = shifted.sum(axis=(1, 2))
    scores = []
    for text in candidates:
        stack, sums, shapes = _templates(text, size)
        if not any(abs(h - crop.shape[0]) <= 1 and abs(w - crop.shape[1]) <= 1 for h, w in shapes):
            continue
        d = np.abs(stack[None] - shifted[:, None]).sum(axis=(2, 3)) / (sums[None] + osum[:, None])
        scores.append((float(d.min()), text))
    if not scores:
        return None, 1.0, 0.0
    scores.sort()
    margin = (scores[1][0] - scores[0][0]) if len(scores) > 1 else 1.0
    return scores[0][1], scores[0][0], margin


def _eq(a, c):
    return (a[..., 0] == c[0]) & (a[..., 1] == c[1]) & (a[..., 2] == c[2])


def _longest_runs(mat):
    """Longest run of True down each column of a 2-D bool array."""
    m = mat.astype(np.int32)
    run = np.zeros(m.shape[1], np.int32)
    best = np.zeros(m.shape[1], np.int32)
    for row in m:
        run = (run + 1) * row
        np.maximum(best, run, out=best)
    return best


def _line_positions(mat):
    """x positions (relative to the panel box) of interior grid lines in `mat` (rows x cols)."""
    idx = [j for j in np.flatnonzero(_longest_runs(mat) >= 6) if 0 < j < mat.shape[1] - 1]
    groups = []
    for j in idx:
        if groups and j - groups[-1][-1] <= 1:
            groups[-1].append(j)
        else:
            groups.append([j])
    return [sum(gp) / len(gp) for gp in groups]


def decode_image(image):
    """Image -> list of panels with packet geometry, tick numbers, title and punch cells."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    need(a.shape[:2] == (CANVAS[1], CANVAS[0]), f"image size {a.shape[1]}x{a.shape[0]} != {CANVAS}")
    sheet, grid, edge, red, white = (_eq(a, c) for c in (SHEET, GRID, EDGE, RED, WHITE))
    paper = sheet | grid | edge | red
    # text ink: blends of TEXT toward white
    t = (255 - a[..., 0]) / (255.0 - TEXT[0])
    pred = np.stack([255 - (255 - TEXT[0]) * t, 255 - (255 - TEXT[1]) * t, 255 - (255 - TEXT[2]) * t], -1)
    ink = ~paper & ~white
    need(np.all(np.abs(pred - a)[ink] <= 2.0), "image: pixels of an undeclared colour (neither palette nor text)")
    alpha = np.where(ink, np.clip(t, 0, 1), 0.0)

    lab, k = ndimage.label(paper, structure=np.ones((3, 3)))
    panels = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) < 2000:
            raise Problem("image: stray paper-coloured mark outside the panels")
        panels.append(dict(label=i, x0=xs.start, y0=ys.start, x1=xs.stop - 1, y1=ys.stop - 1))
    need(panels, "image: no panel")

    for p in panels:
        x0, y0, x1, y1 = p["x0"], p["y0"], p["x1"], p["y1"]
        g = grid[y0:y1 + 1, x0:x1 + 1]

        vx, hy = _line_positions(g), _line_positions(g.T)   # interior grid lines
        w, h = len(vx) + 1, len(hy) + 1
        cw, ch = (x1 - x0) / w, (y1 - y0) / h
        need(abs(cw - ch) <= 1.0, f"image: panel cells not square ({cw:.1f}x{ch:.1f})")
        cell = (cw + ch) / 2
        need(cell >= MIN_CELL_PX, f"image: packet cell {cell:.1f}px < {MIN_CELL_PX}")
        need(all(abs(v - (j + 1) * cw) <= 1.5 for j, v in enumerate(vx)), "image: uneven vertical grid")
        need(all(abs(v - (j + 1) * ch) <= 1.5 for j, v in enumerate(hy)), "image: uneven horizontal grid")
        p.update(w=w, h=h, cell=cell, cw=cw, ch=ch)

        # outline shape from quadrant samples of every cell
        def is_paper(u, v):  # u, v in cells from the panel's top-left corner
            X, Y = int(round(x0 + u * cw)), int(round(y0 + v * ch))
            if paper[Y, X]:
                return True
            need(white[Y, X], "image: unexpected colour inside a panel box")
            return False
        fits = {}
        for shape in ("rect", "d", "a"):
            ok = True
            for r in range(h):
                for c in range(w):
                    for du, dv in ((-.3, -.3), (.3, -.3), (-.3, .3), (.3, .3)):
                        u, v = c + .5 + du, r + .5 + dv
                        if shape == "d":
                            s = v - u
                        elif shape == "a":
                            s = u + v - w
                        else:
                            s = 1.0
                        if abs(s) < 0.15:
                            continue
                        if is_paper(u, v) != (s > 0):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            fits[shape] = ok
        good = [s for s, ok in fits.items() if ok]
        need(len(good) == 1, f"image: panel outline is not one rectangle/triangle ({good})")
        need(good[0] == "rect" or w == h, "image: triangular panel is not square")
        p["shape"] = good[0]

    # ink groups: merge glyphs of one label (horizontal gaps <= 3 px)
    mask = alpha > _INK_THR
    glab, gk = ndimage.label(ndimage.binary_dilation(mask, structure=np.ones((1, 4))), structure=np.ones((3, 3)))
    groups = []
    for sl in ndimage.find_objects(glab):
        ys, xs = sl
        sub = np.where(mask[ys, xs], alpha[ys, xs], 0.0)
        if not sub.any():
            continue
        yy, xx = np.nonzero(sub > _INK_THR)
        groups.append(dict(x0=xs.start + xx.min(), x1=xs.start + xx.max(), y0=ys.start + yy.min(),
                           y1=ys.start + yy.max(), patch=alpha[ys.start + yy.min():ys.start + yy.max() + 1,
                                                               xs.start + xx.min():xs.start + xx.max() + 1]))
    nums = [str(v) for v in range(1, 21)]
    titles = []
    for gr in groups:
        owner = None
        for p in panels:
            if p["y0"] - 17 <= gr["y0"] and gr["y1"] < p["y0"] and p["x0"] - 6 <= gr["x0"] and gr["x1"] <= p["x1"] + 6:
                owner = (p, "top")
            elif p["x0"] - 23 <= gr["x0"] and gr["x1"] < p["x0"] and p["y0"] - 6 <= gr["y0"] and gr["y1"] <= p["y1"] + 6:
                owner = (p, "left")
        if owner is None:
            titles.append(gr)
            continue
        p, side = owner
        text, dist, margin = ocr(gr["patch"], nums, 9)
        need(text is not None and dist <= OCR_MAX_DIST and margin >= OCR_MIN_MARGIN,
             f"image: illegible tick label at ({gr['x0']},{gr['y0']}) (best={text}, d={dist:.2f}, margin={margin:.2f})")
        p.setdefault("ocr", []).append((dist, margin))
        if side == "top":
            c = int(round((gr["x0"] + 4 - p["x0"]) / p["cw"] - 0.5))
            p.setdefault("cols", {})
            need(0 <= c < p["w"] and c not in p["cols"], "image: tick label not over a single column")
            p["cols"][c] = int(text)
        else:
            r = int(round(((gr["y0"] + gr["y1"]) / 2 - p["y0"]) / p["ch"] - 0.5))
            p.setdefault("rows", {})
            need(0 <= r < p["h"] and r not in p["rows"], "image: tick label not beside a single row")
            p["rows"][r] = int(text)

    for p in panels:
        for key, size in (("cols", "w"), ("rows", "h")):
            ticks = p.get(key, {})
            # cells >= 12 px: every row and column of a packet carries its original-sheet number
            need(sorted(ticks) == list(range(p[size])), f"image: {key} tick numbers missing ({sorted(ticks)})")
            bases = {v - j - 1 for j, v in ticks.items()}
            need(len(bases) == 1, f"image: {key} tick numbers not consecutive ({sorted(ticks.items())})")
            p["base_" + key] = bases.pop()
            need(p["base_" + key] >= 0, "image: tick number below 1")

    # titles P1..Pk, each up-left of exactly one panel
    for gr in titles:
        text, dist, margin = ocr(gr["patch"], [f"P{i}" for i in range(1, 7)], 11)
        need(text is not None and dist <= OCR_MAX_DIST and margin >= OCR_MIN_MARGIN,
             f"image: unexpected text at ({gr['x0']},{gr['y0']}) (best={text}, d={dist:.2f})")
        best = None
        for p in panels:
            dx, dy = p["x0"] - gr["x1"], p["y0"] - gr["y1"]
            if 0 < dx <= 120 and 0 < dy <= 120 and (best is None or dx + dy < best[0]):
                best = (dx + dy, p)
        need(best is not None, f"image: title {text} not attached to a panel")
        need("title" not in best[1], "image: two titles on one panel")
        best[1]["title"] = int(text[1:])
        best[1]["title_ocr"] = (dist, margin)
    need(all("title" in p for p in panels), "image: panel without a P-title")
    need(sorted(p["title"] for p in panels) == list(range(1, len(panels) + 1)), "image: titles are not P1..Pk")

    # red punch dots
    rlab, _rk = ndimage.label(red, structure=np.ones((3, 3)))
    for j, sl in enumerate(ndimage.find_objects(rlab), 1):
        comp = rlab[sl] == j
        area = int(comp.sum())
        cy, cx = ndimage.center_of_mass(comp)
        cy += sl[0].start
        cx += sl[1].start
        radius = (area / np.pi) ** 0.5
        need(radius >= MIN_DOT_RADIUS_PX, f"image: punch dot too small (r={radius:.1f}px)")
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        need(abs(bh - bw) <= 1 and abs(bh - 2 * radius) <= 2.0, "image: red mark is not a round dot")
        owner = [p for p in panels if p["x0"] < cx < p["x1"] and p["y0"] < cy < p["y1"]]
        need(len(owner) == 1, "image: red mark outside every panel")
        p = owner[0]
        fu, fv = (cx - p["x0"]) / p["cw"], (cy - p["y0"]) / p["ch"]
        c, r = int(fu), int(fv)
        off = max(abs(fu - c - .5), abs(fv - r - .5))
        need(off <= MAX_DOT_OFFSET, f"image: dot {off:.2f} cell from a cell centre (ambiguous cell)")
        need(radius / p["cell"] <= MAX_DOT_RADIUS, "image: dot larger than its cell allows")
        u, v = c + .5, r + .5
        if p["shape"] == "d":
            need(v - u >= 1, "image: dot on or beyond the fold diagonal")
        elif p["shape"] == "a":
            need(u + v - p["w"] >= 1, "image: dot on or beyond the fold diagonal")
        ring = [(cx + (radius + 2) * np.cos(t), cy + (radius + 2) * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        need(all(sheet[int(round(y)), int(round(x))] for x, y in ring), "image: dot not surrounded by paper")
        X, Y = 2 * (p["base_cols"] + c) + 1, 2 * (p["base_rows"] + r) + 1
        p.setdefault("dots", set())
        need((X, Y) not in p["dots"], "image: two dots in one cell")
        p["dots"].add((X, Y))
        p.setdefault("dot_stats", []).append((off, radius))
        p.setdefault("dot_discs", []).append((cx, cy, radius))

    masks = dict(sheet=sheet, grid=grid, edge=edge, red=red, white=white, other=ink)
    for p in panels:
        bad = unexplained_pixels(p, masks)
        need(bad == 0, f"image: {bad} panel pixels not explained by sheet/grid/outline/dots (extra marks)")
    return panels


BAND_PX = 1.6       # outline band half-width
GRID_TOL_PX = 1.0   # grid pixel to detected grid line
DOT_PAD_PX = 1.5    # red pixel to decoded dot disc


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def unexplained_pixels(p, masks) -> int:
    """Pixel accounting for one decoded panel: every pixel of its box must be sheet
    inside the outline, grid on a detected grid line, outline colour on the outline,
    red inside a decoded dot, or white outside the outline; and every detected
    interior grid line must be drawn across the whole packet.  Returns the number of
    pixels that break this (0 for every rendered item)."""
    x0, y0, x1, y1 = p["x0"], p["y0"], p["x1"], p["y1"]
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(float)
    verts = {"rect": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "d": [(x0, y0), (x1, y1), (x0, y1)],
             "a": [(x1, y0), (x1, y1), (x0, y1)]}[p["shape"]]
    band = np.min([_seg_dist(xs, ys, *verts[i], *verts[(i + 1) % len(verts)]) for i in range(len(verts))],
                  axis=0) <= BAND_PX
    if p["shape"] == "rect":
        inside = np.ones_like(band)
    elif p["shape"] == "d":
        inside = (ys - y0) * (x1 - x0) >= (xs - x0) * (y1 - y0)
    else:
        inside = (ys - y0) * (x1 - x0) >= (x1 - xs) * (y1 - y0)
    sub = {k: m[y0:y1 + 1, x0:x1 + 1] for k, m in masks.items()}
    vx = [x0 + v for v in _line_positions(sub["grid"])]
    hy = [y0 + v for v in _line_positions(sub["grid"].T)]
    on_grid = np.zeros_like(band)
    for v in vx:
        on_grid |= np.abs(xs - v) <= GRID_TOL_PX
    for v in hy:
        on_grid |= np.abs(ys - v) <= GRID_TOL_PX
    in_dot = np.zeros_like(band)
    for cx, cy, rad in p.get("dot_discs", ()):
        in_dot |= np.hypot(xs - cx, ys - cy) <= rad + DOT_PAD_PX
    bad = (sub["other"]
           | (sub["sheet"] & ~(inside | band))
           | (sub["white"] & inside & ~band)
           | (sub["edge"] & ~band)
           | (sub["grid"] & ~((on_grid & inside) | band))
           | (sub["red"] & ~in_dot))
    # an interior grid line must be present in every row (column) of the packet interior
    drawn = sub["grid"] | sub["edge"]
    req = inside & ~band & ~in_dot
    for v in vx:
        cols = np.abs(xs[0] - v) <= GRID_TOL_PX
        bad[:, np.flatnonzero(cols)[0]] |= req[:, cols].all(axis=1) & ~drawn[:, cols].any(axis=1)
    for v in hy:
        rws = np.abs(ys[:, 0] - v) <= GRID_TOL_PX
        bad[np.flatnonzero(rws)[0], :] |= req[rws, :].all(axis=0) & ~drawn[rws, :].any(axis=0)
    return int(bad.sum())


# --------------------------------------------------------------------------- record

def _check(record, image):
    tier = record.get("tier")
    need(tier in TIERS, f"unknown tier {tier!r}")
    cfg = TIERS[tier]
    m = record["meta"]
    need(record.get("answer_type") == "bitstring", "answer_type is not bitstring")
    need(record.get("choices") is None, "bitstring item carries choices")
    need(m.get("question_form") == "staged_paper_punches/bitstring", "meta.question_form")
    need(m.get("template") == cfg["template"], f"meta.template {m.get('template')!r} != {cfg['template']!r} for {tier}")

    # 1. text
    q = record["question"]
    n, pfolds, stages, queries = parse_question(q)
    folds = resolve_folds(n, pfolds)
    need(emit_question(cfg["template"], n, pfolds, stages, queries) == q,
         "question text differs from the declared template")

    # 2. tier conformance (from text)
    need(n == cfg["n"], f"sheet {n} != tier n {cfg['n']}")
    need(len(folds) == cfg["folds"], f"{len(folds)} folds != tier {cfg['folds']}")
    need(stages == cfg["stages"], f"punch stages {stages} != tier {cfg['stages']}")
    need(len(queries) == cfg["queries"], f"{len(queries)} queries != tier {cfg['queries']}")
    diag = [f["axis"] in "da" for f in folds]
    need(diag == [False] * (len(folds) - 1) + [cfg["diagonal_last"]], "diagonal-fold pattern does not match the tier")
    need(len(set(queries)) == len(queries), "a cell is queried twice")
    need(all(0 <= r < n and 0 <= c < n for r, c in queries), "query outside the sheet")

    # 3. image
    panels = decode_image(image)
    need(len(panels) == len(stages), f"{len(panels)} panels in the image, text names {len(stages)}")
    punches = {}
    for p in panels:
        stage = stages[p["title"] - 1]
        f = folds[stage - 1]
        x0, y0, x1, y1 = f["after"]
        need((p["w"], p["h"]) == ((x1 - x0) // 2, (y1 - y0) // 2),
             f"P{p['title']}: packet {p['w']}x{p['h']} cells, text implies {(x1 - x0) // 2}x{(y1 - y0) // 2}")
        need((2 * p["base_cols"], 2 * p["base_rows"]) == (x0, y0),
             f"P{p['title']}: tick numbers place the packet at col {p['base_cols'] + 1}, row {p['base_rows'] + 1}; "
             f"text implies col {x0 // 2 + 1}, row {y0 // 2 + 1}")
        need(p["shape"] == ("rect" if f["axis"] in "vh" else f["axis"]), f"P{p['title']}: outline shape vs fold {stage}")
        dots = p.get("dots", set())
        need(len(dots) == cfg["punches"][p["title"] - 1], f"P{p['title']}: {len(dots)} dots, tier needs "
                                                           f"{cfg['punches'][p['title'] - 1]}")
        need(all(legal_cell(X, Y, f) for X, Y in dots), f"P{p['title']}: dot off the packet")
        punches[stage] = dots

    # 4. simulate and compare
    holes, effects = simulate(n, folds, punches)
    gold = "".join("1" if q_ in holes else "0" for q_ in queries)
    need(re.fullmatch(f"[01]{{{len(queries)}}}", record["answer"] or ""), "answer is not a fixed-length bitstring")
    need(record["answer"] == gold, f"answer {record['answer']} != recomputed {gold}")
    qset = set(queries)
    for s, e in effects.items():
        own = e - set().union(*(o for t, o in effects.items() if t != s))
        need(own, f"punch stage after fold {s} is subsumed by the others")
        need(own & qset, f"no queried cell depends on the punch stage after fold {s}")

    # 5. no text-only shortcut through the query list, and no answer-shaped example
    sig = query_signatures(n, folds, stages, queries)
    ones = collections.Counter(s for s, b in zip(sig, gold) if b == "1")
    zeros = collections.Counter(s for s, b in zip(sig, gold) if b == "0")
    need(ones == zeros, "query list is not structure-balanced: the 1-cells and 0-cells differ in mate/crease "
                        "signatures that the fold text alone reveals")
    for problem in example_overlap(q, gold):
        need(False, problem)

    # 6. meta cross-check
    need(m.get("n") == n, "meta.n")
    need([{k: f[k] for k in ("axis", "line", "keep", "before", "after")} for f in m["folds"]] == folds,
         "meta.folds disagree with the question text")
    need([e["stage"] for e in m["events"]] == stages, "meta.events stages")
    need(all({tuple(c) for c in e["centers"]} == punches[e["stage"]] and len(e["centers"]) == len(punches[e["stage"]])
             for e in m["events"]), "meta.events disagree with the image punches")
    need(m["truth"] == [[int((r, c) in holes) for c in range(n)] for r in range(n)], "meta.truth disagrees with simulation")
    need([tuple(x) for x in m["queries"]] == queries, "meta.queries disagree with the question text")
    tp = m.get("tier_params") or {}
    need({k: tp.get(k) for k in ("n", "folds", "stages", "punches", "queries", "diagonal_last")} ==
         {k: cfg[k] for k in ("n", "folds", "stages", "punches", "queries", "diagonal_last")}, "meta.tier_params")
    # the dedup key is the digest of the verified folds and image punches (queries excluded)
    core = dict(n=n, folds=[[f["axis"], f["line"], f["keep"]] for f in folds],
                events=[[s, sorted(map(list, punches[s]))] for s in stages])
    need(record.get("core_key") == digest(dict(task="paper_folding", core=core)),
         "core_key is not the digest of the verified folds and punches")
    return dict(holes=len(holes), panels=panels, signatures=sig)


def verify(record: dict, image) -> list[str]:
    try:
        _check(record, image)
    except Problem as exc:
        return [str(exc)]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return [f"malformed record: {exc!r}"]
    return []
