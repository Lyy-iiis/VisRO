"""Independent verifier for mirroring_patterns.

Imports nothing from the generator (generator/, shared/puzzles) or from the adapter.
Everything is recomputed from the QUESTION TEXT and the IMAGE PIXELS:

  text   regex-parse n, the step lines ("a–b×c–d rows|columns", one per
         operation, in order), the queried cells and the requested length;
         rebuild the declared prompt from those parts and require byte
         equality (so every convention sentence is present and nothing else
         is);
  image  locate the grid from its 1-px outline lines, require n x n equal
         cells, read each cell's WHOLE interior (must be one uniform colour)
         and name it by hue/saturation/value -- not by the generator's RGB
         table; OCR the row labels (left) and column labels (top) against
         glyphs rendered from the committed DejaVuSans.ttf and require 1..n;
         require that nothing else is drawn outside the grid;
  solve  forward-simulate the reflections on the decoded grid with numpy
         index arithmetic (the generator traces each query BACKWARDS), read the
         queried cells, compare with the gold answer;
  rules  the tier's declared generation rules: (n, steps, nq), alternating
         axes, operation 1 reverses rows at hard and above, every rectangle
         straddles the grid centre, queries in row-major order at easy/medium,
         minimum hits, source != query, answer != initial reading, no
         whole-grid symmetry; and no answer-shaped letter string (a format
         example) anywhere in the question;
  meta   meta.grid / final_grid / operations / queries / requirements must equal
         what was decoded and simulated, and core_key must equal an
         independently computed D4-canonical digest of the answer-relevant
         content (operations, queried cells and their final letters).
"""
from __future__ import annotations

import colorsys
import functools
import re

import numpy as np

from visro.paths import ROOT
from visro.seeding import canonical, digest

CORRECTNESS = ("The gold string is recomputed by forward-simulating the operations parsed from the question text on "
               "the grid decoded from the PNG (grid lines, uniform cell interiors named by hue, OCR'd 1..n row/column "
               "labels), then cross-checked against meta.grid/final_grid/operations/queries; no generator code is used.")
UNIQUENESS = ("There are no options: once the prompt fixes (row,column) 1-based coordinates, inclusive ranges, "
              "in-order application on the current grid and one letter per cell (W white), the answer is a function of "
              "the image; the verifier requires that exact prompt, an unambiguous colour name for every cell, labels "
              "reading 1..n, and nothing drawn outside the grid.")

TIERS = {  # tier -> (n, steps, nq, minimum_hits); restated here, not imported
    "easy": (4, 1, 3, 1), "medium": (6, 2, 3, 1), "hard": (8, 2, 4, 2),
    "xhard": (10, 4, 8, 3), "xxhard": (12, 6, 12, 4),
}
ROW_MAJOR_TIERS = ("easy", "medium")           # queries listed in row-major order
FIRST_ROWS_TIERS = ("hard", "xhard", "xxhard")  # operation 1 reverses rows
ALPHABET = "RBGYPW"
ANSWER_SHAPED = re.compile(f"[{ALPHABET}]{{2,}}")  # a run of >= 2 answer letters = a would-be format example
LEGEND = "R red, B blue, G green, Y yellow, P purple, W white"
NAME2CODE = {"red": "R", "blue": "B", "green": "G", "yellow": "Y", "purple": "P", "white": "W"}
FONT = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"

# compact prompt: line 1 = conventions, then one "a–b×c–d rows|columns" line per step, then the
# query line.  Restated here, not imported.
N_RE = re.compile(r"^Cells are \(row,column\), numbered 1–(\d+) as labelled\. ")
OP_RE = re.compile(r"^(\d+)–(\d+)×(\d+)–(\d+) (rows|columns)$")
Q_RE = re.compile(r"^Then give the colors of ((?:\(\d+,\d+\),?)+) in order as (\d+) letters")


class Problem(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Problem(msg)


# --------------------------------------------------------------------------- text

def expected_text(n, ops, queries):
    """The declared prompt, rebuilt from parsed parts (ops: (axis, a, b, c, d) 1-based inclusive, axis
    "rows" or "columns"): (row,column) cells numbered 1..n as labelled, steps applied in order each to the
    previous result, row/column ORDER reversed only inside the rows×columns rectangle, the listed cells read
    afterwards in the listed order as N letters with no separators, and the colour legend."""
    lines = [f"{a}–{b}×{c}–{d} {ax}" for ax, a, b, c, d in ops]
    cells = ",".join(f"({r},{c})" for r, c in queries)
    lead = ("Apply this step" if len(ops) == 1 else "Apply each step in order to the previous result")
    return (f"Cells are (row,column), numbered 1–{n} as labelled. {lead}: reverse the order of the given rows or "
            "columns inside the rows×columns rectangle.\n" + "\n".join(lines) + "\n"
            f"Then give the colors of {cells} in order as {len(queries)} letters, no separators: {LEGEND}.")


def parse_question(q):
    m = N_RE.match(q)
    need(m, "question does not state an n×n grid")
    n = int(m.group(1))
    lines = q.split("\n")
    need(len(lines) >= 3, "question is not conventions / step lines / query line")
    ops = []
    for k, line in enumerate(lines[1:-1], 1):
        mo = OP_RE.match(line)
        need(mo, f"step line {k} is not 'a–b×c–d rows|columns'")
        a, b, c, d, what = mo.groups()
        a, b, c, d = map(int, (a, b, c, d))
        need(1 <= a < b <= n and 1 <= c < d <= n, f"operation {k} rectangle outside the grid or degenerate")
        ops.append((what, a, b, c, d))
    mq = Q_RE.match(lines[-1])
    need(mq, "query sentence not found")
    queries = [(int(r), int(c)) for r, c in re.findall(r"\((\d+),(\d+)\)", mq.group(1))]
    need(int(mq.group(2)) == len(queries), "stated answer length != number of queried cells")
    need(len(set(queries)) == len(queries), "a cell is queried twice")
    need(all(1 <= r <= n and 1 <= c <= n for r, c in queries), "queried cell outside the grid")
    m_ex = ANSWER_SHAPED.search(q)
    need(m_ex is None, f"question contains an answer-shaped letter string {m_ex and m_ex.group(0)!r} (format example)")
    need(q == expected_text(n, ops, queries), "question text differs from the declared form")
    return n, ops, queries


# --------------------------------------------------------------------------- image

def colour_name(rgb):
    """Colour NAME with a safety margin, or None when it is not unambiguous."""
    r, g, b = (float(v) / 255 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.06 and v > 0.93:
        return "white"
    if s < 0.35 or v < 0.35:
        return None
    deg = h * 360
    for name, lo, hi in (("red", -25, 12), ("yellow", 38, 62), ("green", 120, 170),
                         ("blue", 200, 235), ("purple", 265, 300)):
        d = deg - 360 if deg > 300 else deg
        if lo <= d <= hi:
            return name
    return None


def _runs(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return [int(idx[0])] + [int(b) for a, b in zip(idx[:-1], idx[1:]) if b != a + 1]


@functools.lru_cache(maxsize=None)
def _glyph(text, fx, fy):
    """Darkness map (0..1) of `text` in DejaVuSans 12px, cropped to its ink."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), 12)
    im = Image.new("L", (48, 32), 255)
    ImageDraw.Draw(im).text((6 + fx, 6 + fy), text, fill=0, font=font)
    a = (255 - np.asarray(im, dtype=float)) / 255
    ys, xs = np.nonzero(a > 0.15)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


_CANVAS = (18, 28)  # label crops are placed at (2, 2); templates at (2+dy, 2+dx), dy, dx in -1..1


@functools.lru_cache(maxsize=None)
def _bank(top):
    """Every candidate label "0".."top" at both sub-pixel phases and ±1 px shifts, as one array."""
    arrs, texts = [], []
    for k in range(top + 1):
        for fx in (0.0, 0.5):
            for fy in (0.0, 0.5):
                t = _glyph(str(k), fx, fy)
                t = t / t.max()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        b = np.zeros(_CANVAS, dtype=np.float32)
                        b[2 + dy:2 + dy + t.shape[0], 2 + dx:2 + dx + t.shape[1]] = t
                        arrs.append(b)
                        texts.append(str(k))
    bank = np.stack(arrs)
    return bank, bank.sum(axis=(1, 2)), np.array(texts)


def read_label(dark, box, n):
    """OCR the label inside box=(x0,y0,x1,y1) of the darkness map; returns (text, bbox).

    Score per candidate = min over phases/shifts of sum|crop - glyph| / max(ink); the best
    candidate must be clearly better than every other number in 0..n+1."""
    x0, y0, x1, y1 = box
    sub = dark[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub > 0.15)
    need(ys.size > 0, f"missing label in {box}")
    crop = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    need(crop.shape[0] <= _CANVAS[0] - 4 and crop.shape[1] <= _CANVAS[1] - 4, f"label in {box} is too large")
    a = np.zeros(_CANVAS, dtype=np.float32)
    a[2:2 + crop.shape[0], 2:2 + crop.shape[1]] = crop / max(crop.max(), 1e-9)
    bank, ink, texts = _bank(n + 1)
    raw = np.abs(bank - a).sum(axis=(1, 2)) / np.maximum(ink, a.sum())
    scores = sorted((float(raw[texts == t].min()), t) for t in dict.fromkeys(texts.tolist()))
    need(scores[0][0] < 0.35 and scores[1][0] - scores[0][0] > 0.12,
         f"illegible label in {box}: best {scores[0]}, next {scores[1]}")
    ys, xs = np.nonzero(sub > 0.02)  # full anti-aliased extent, for the stray-ink check
    return scores[0][1], (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


def decode_image(im, n_expected):
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    H, W, _ = a.shape
    need((W, H) == (448, 448), f"image size {(W, H)} != 448x448")
    flat = a.reshape(-1, 3)
    spread = flat.max(1) - flat.min(1)
    grey = flat[(spread < 20) & (flat.max(1) < 200) & (flat.max(1) > 60)]
    need(len(grey) > 0, "no grid outline found")
    vals, counts = np.unique(grey, axis=0, return_counts=True)
    outline = vals[counts.argmax()]
    is_line = np.all(a == outline, axis=2)
    cols, rows = is_line.sum(0), is_line.sum(1)
    xs, ys = _runs(cols > 0.5 * cols.max()), _runs(rows > 0.5 * rows.max())
    need(len(xs) == len(ys) == n_expected + 1, f"found {len(xs) - 1}x{len(ys) - 1} grid cells, question says {n_expected}")
    n = n_expected
    dx, dy = np.diff(xs), np.diff(ys)
    need(dx.min() == dx.max() == dy.min() == dy.max(), "cells are not equal squares")
    cell = int(dx[0])
    need(cell >= 20, f"cells too small to read ({cell} px)")
    gx0, gy0, gx1, gy1 = xs[0], ys[0], xs[-1], ys[-1]
    # every grid line must be a full-length 1-px outline
    need(is_line[gy0:gy1 + 1, xs].all() and is_line[ys, gx0:gx1 + 1].all(), "broken grid line")
    grid = []
    for r in range(n):
        row = ""
        for c in range(n):
            inner = a[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]].reshape(-1, 3)
            need((inner == inner[0]).all(), f"cell ({r + 1},{c + 1}) interior is not one uniform colour")
            name = colour_name(inner[0])
            need(name is not None, f"cell ({r + 1},{c + 1}) colour {tuple(int(v) for v in inner[0])} is not an unambiguous named colour")
            row += NAME2CODE[name]
        grid.append(row)
    # labels: nothing but row labels left of the grid and column labels above it
    dark = (255 - a.min(axis=2)).astype(float) / 255
    boxes = []
    for k in range(n):
        text, bb = read_label(dark, (0, int(ys[k]) + 1, int(gx0) - 1, int(ys[k + 1])), n)
        need(text == str(k + 1), f"row {k + 1} is labelled {text!r}")
        need(abs((bb[1] + bb[3]) / 2 - (ys[k] + ys[k + 1]) / 2) <= cell / 3, f"row label {k + 1} is off its row")
        boxes.append(bb)
        text, bb = read_label(dark, (int(xs[k]) + 1, 0, int(xs[k + 1]), int(gy0) - 1), n)
        need(text == str(k + 1), f"column {k + 1} is labelled {text!r}")
        need(abs((bb[0] + bb[2]) / 2 - (xs[k] + xs[k + 1]) / 2) <= cell / 3, f"column label {k + 1} is off its column")
        boxes.append(bb)
    ink = dark > 0.02
    ink[gy0:gy1 + 1, gx0:gx1 + 1] = False
    for x0, y0, x1, y1 in boxes:
        ink[y0:y1, x0:x1] = False
    need(not ink.any(), f"{int(ink.sum())} stray non-white pixels outside the grid and its labels")
    return grid, dict(x0=int(gx0), y0=int(gy0), cell=cell)


# --------------------------------------------------------------------------- solve

def simulate(grid, ops):
    """Forward simulation: where does every original cell end up?"""
    n = len(grid)
    pr, pc = np.divmod(np.arange(n * n), n)
    hits = np.zeros(n * n, dtype=int)
    for what, a, b, c, d in ops:
        a, b, c, d = a - 1, b - 1, c - 1, d - 1
        inside = (pr >= a) & (pr <= b) & (pc >= c) & (pc <= d)
        hits += inside
        if what == "rows":
            pr = np.where(inside, a + b - pr, pr)
        else:
            pc = np.where(inside, c + d - pc, pc)
    src = np.full((n, n), -1)
    src[pr, pc] = np.arange(n * n)
    need((src >= 0).all(), "operations are not a permutation")
    letters = np.array([list(row) for row in grid])
    final = letters.reshape(-1)[src]
    return ["".join(row) for row in final.tolist()], src, hits


def canonical_core(n, ops, queries, final):
    """D4-canonical digest of the ANSWER-RELEVANT content -- the operations, and every queried cell
    with the letter it ends up with -- by explicit coordinate maps.  Cells no query reads do not
    enter, so items with the same question and answer over otherwise different grids collide."""
    forms = []
    for t in range(8):
        def f(r, c, t=t):
            if t & 1:
                r, c = c, r
            if t & 2:
                r = n - 1 - r
            if t & 4:
                c = n - 1 - c
            return r, c
        o = []
        for what, a, b, c, d in ops:
            corners = [f(a - 1, c - 1), f(b - 1, d - 1)]
            r0, r1 = sorted(p[0] for p in corners)
            c0, c1 = sorted(p[1] for p in corners)
            axis = 0 if what == "rows" else 1
            if t & 1:
                axis = 1 - axis
            o.append([axis, r0, r1 + 1, c0, c1 + 1])
        cells = sorted([*f(r - 1, c - 1), final[r - 1][c - 1]] for r, c in queries)
        forms.append(canonical([o, cells]))
    return digest({"task": "mirroring_patterns", "n": n, "answer_relevant_core": min(forms)})


# --------------------------------------------------------------------------- verify

def _verify(rec, image):
    need(rec.get("answer_type") == "text" and not rec.get("choices"), "declared form must be a free-text code string")
    tier = rec.get("tier")
    need(tier in TIERS, f"unknown tier {tier!r}")
    n_t, steps_t, nq_t, hits_t = TIERS[tier]
    n, ops, queries = parse_question(rec["question"])
    need((n, len(ops), len(queries)) == (n_t, steps_t, nq_t),
         f"{tier} must be n={n_t}, {steps_t} operations, {nq_t} queries; got n={n}, {len(ops)}, {len(queries)}")
    need(all(ops[i][0] != ops[i + 1][0] for i in range(len(ops) - 1)), "operations must alternate axes")
    need(tier not in FIRST_ROWS_TIERS or ops[0][0] == "rows", f"{tier}: operation 1 must reverse rows")
    need(all(a - 1 < n / 2 < b and c - 1 < n / 2 < d for _, a, b, c, d in ops),
         "every rectangle must straddle the grid centre (r0,c0 < n/2 < r1,c1, 0-based half-open)")
    need(tier not in ROW_MAJOR_TIERS or queries == sorted(queries), f"{tier}: queries must be in row-major order")
    answer = rec["answer"]
    need(isinstance(answer, str) and re.fullmatch(f"[RBGYPW]{{{nq_t}}}", answer) is not None,
         f"answer {answer!r} is not {nq_t} letters from RBGYPW")
    need(answer not in rec["question"], "the gold string appears in the question (format example)")
    grid, geom = decode_image(image, n)
    need(geom["cell"] == 352 // n, f"cell size {geom['cell']} px")
    final, src, hits = simulate(grid, ops)
    gold = "".join(final[r - 1][c - 1] for r, c in queries)
    need(answer == gold, f"answer {answer!r} != recomputed {gold!r}")
    # declared generation rules
    need(answer != "".join(grid[r - 1][c - 1] for r, c in queries), "answer equals the initial-grid reading")
    for r, c in queries:
        s = int(src[r - 1, c - 1])
        need(s != (r - 1) * n + (c - 1), f"queried cell ({r},{c}) reads itself")
        need(hits[s] >= hits_t, f"queried cell ({r},{c}) passes through {hits[s]} < {hits_t} rectangles")
    perm = src.reshape(-1)
    ident = np.arange(n * n).reshape(n, n)
    need(not any(np.array_equal(perm, s.reshape(-1)) for k in range(4)
                 for s in (np.rot90(ident, k), np.fliplr(np.rot90(ident, k)))),
         "operations compose to a whole-grid symmetry")
    # meta must describe exactly this item
    m = rec.get("meta") or {}
    need(m.get("n") == n, "meta.n")
    need(m.get("grid") == grid, "meta.grid differs from the image")
    need(m.get("final_grid") == final, "meta.final_grid differs from the forward simulation")
    need(m.get("operations") == [{"axis": "rows" if w == "rows" else "cols", "rows": [a, b], "cols": [c, d]}
                                 for w, a, b, c, d in ops], "meta.operations differ from the question")
    need(m.get("queries") == [[r, c] for r, c in queries], "meta.queries differ from the question")
    need(m.get("requirements") == {"n": n_t, "steps": steps_t, "nq": nq_t, "minimum_hits": hits_t}, "meta.requirements")
    need(m.get("minimum_query_operations") == hits_t, "meta.minimum_query_operations")
    need(rec.get("core_key") == canonical_core(n, ops, queries, final),
         "core_key is not the D4-canonical answer-relevant digest")


def verify(record, image):
    try:
        _verify(record, image)
    except Problem as exc:
        return [str(exc)]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return [f"malformed record: {exc!r}"]
    return []
