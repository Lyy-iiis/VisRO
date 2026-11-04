"""Independent verifier for logic_matrix.

Imports nothing from the generator (generator/, shared/puzzles) or from the
adapter.  Everything below is written from the rules the QUESTION TEXT states
and from the published page layout, and is checked against the PNG pixels:

  question   must equal, character for character, the declared form rebuilt
             from the parameters parsed out of the question itself; those
             parameters must agree with meta.  No answer-shaped example may
             appear in it.
  image      every grid / tile / glyph the solver needs is DECODED from the
             pixels at the published layout (re-derived here, and compared to
             meta.layout where the renderer published one), labels are matched
             against rendered templates, the answer cell must be blank, and
             nothing may be drawn outside the known elements.  Whole cells and
             whole tiles are checked, not only their sample points.
  solve      boolean tiers are solved from the decoded image grids and the
             question's steps alone; overlay from the decoded masks; attribute
             from meta glyphs that the decode proved equal to the image.
  unique     the full hypothesis space the question declares is enumerated;
             it must predict exactly one answer, and (multiple choice) exactly
             one option, the gold letter, may match it.  Every distractor must
             also differ from the gold in an attribute that is perceptible at
             the delivered resolution.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import itertools
import math
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CORRECTNESS = (
    "Question must equal the declared template rebuilt from its own parsed parameters (which must match meta) and "
    "carries no answer-shaped example. Boolean tiers: every demonstration/input grid is decoded from cell-centre "
    "pixels at an independently derived layout, every cell interior must be one of exactly two appearances (one "
    "per value), operator and input labels are template-matched, and the answer is recomputed from the decoded "
    "grids and the question's steps; overlay: tile masks are decoded at tile centroids and every eroded tile must "
    "be wholly on-colour or white, and the operation is re-inferred; attribute: each glyph's colour, pip count, "
    "pip-ring angle, outline shape and size are decoded at one page-wide glyph scale and must equal meta (shape "
    "and size decoded jointly), then the rule family is enumerated. The hidden cell may show only the "
    "template-matched '?', and no pixel may be drawn off the expected glyphs, tiles, grids and labels.")
UNIQUENESS = (
    "The complete hypothesis space the question states is enumerated (attribute: every constant/cyclic/stepped "
    "program per attribute with exactly N changing, rotations compared modulo the glyph's symmetry; overlay and "
    "boolean: all 4 operations per operator label, labels allowed to coincide); it must predict exactly one "
    "answer, exactly one option may match it, and every distractor must differ from the gold in shape, pip "
    "count, colour (L1>=60) or rotation (>=15 deg modulo symmetry), never only in size or a smaller rotation.")

TIERS = ("easy", "medium", "hard", "xhard", "xxhard")
IMAGE_SIZE = {"easy": 448, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}
# The label templates are rendered with the repository's sha-pinned DejaVu fonts in
# <repo>/assets/fonts; release/ alone does not carry them.  A missing font is reported as an
# "environment:" problem, never a crash.
FONT_DIR = (_ROOT / 'assets/fonts')

# Truth tables indexed by 2*left+right: OR, AND, XOR, left-minus-right.
TABLES = ((0, 1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0))
SHAPES = ("circle", "triangle", "square", "pentagon", "hexagon", "star")
SIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "star": 5}
SIZES = (0.85, 1.0, 1.15)
ROT_STEPS = (30, 45, 60, 90)
ROW_ROT = 40
ROT_MIN = 15.0      # smallest rotation difference (mod symmetry) accepted as the only visible difference
COLOR_MIN = 60      # smallest L1 colour difference accepted as perceptible
WHITE = (255, 255, 255)

CHOICE_SENTENCE = 'Answer A, B, C or D.'
ATTR_TEMPLATE = (
    'Exactly {n} of five attributes (pip count, color, outline shape, size, rotation) change; the'
    ' rest stay constant. Pips add one per column right and one per row down. Color, shape and '
    'size cycle through three values left to right (size: small, medium, large), each row '
    'starting one or two places later. Rotation adds 30, 45, 60 or 90 degrees clockwise per '
    'column, each row starting 40 degrees further; pips turn with the outline. Size is the '
    'circumradius, at matrix scale. Which option completes the bottom-right cell? '
    + CHOICE_SENTENCE
)
OVERLAY_TEMPLATE = (
    'Filled tiles are 1, empty 0. In both rows the third figure combines the first two by the '
    'same tile-wise operation: OR, AND, XOR or left-minus-right (filled left, empty right). Infer'
    ' it from the first row; which option completes the second row? Only whole tiles count. '
    + CHOICE_SENTENCE
)


def _boolean_question(steps, n):
    """The one boolean template (hard, xhard, xxhard).  steps = [(degrees, label, input index)]."""
    labels = [lab for _, lab, _ in steps]
    named = " and ".join(labels) if len(labels) == 2 else ", ".join(labels[:-1]) + " and " + labels[-1]
    steps = "; ".join(f"{deg} degrees, {lab}, I{inp}" for deg, lab, inp in steps)
    return ("Each label (" + named + ") is one tile-wise operation, the same in both its demonstrations (left, "
            "label, right = result): OR, AND, XOR or left-minus-right; labels may share one. Start from I1; each step "
            "turns the current grid clockwise by its angle, then applies its label with the current grid on the left "
            f"and its input on the right. Steps: {steps}. Answer the final grid as {n} rows of {n} digits (1 filled, "
            "0 empty), top to bottom, separated by /.")


def _format_example_problems(q, answer, kind):
    """No question may print an answer-shaped example (requirement: an example must never equal or
    share a component with a gold).  logic_matrix questions carry no example at all, so ANY
    answer-shaped string is a problem:
      grid    a run of 3+ binary digits, a '/'-joined bit pattern, the gold or any of its rows;
      choice  a standalone option letter outside the declared choice list (the article 'A' before a
              lower-case word is English, not a letter), or a quoted token."""
    probs = []
    if re.search(r"\be\.g\.|\bexamples?\b|\bsuch as\b", q, re.I):
        probs.append("question: contains a worked/format example")
    if re.search(r"[\"“”]", q):
        probs.append("question: contains a quoted token")
    if kind == "grid":
        if re.search(r"[01]{3,}", q) or re.search(r"[01]\s*/\s*[01]", q):
            probs.append("question: contains an answer-shaped bit pattern")
        rows = answer.split("/")
        if answer in q or any(re.search(r"(?<![0-9])%s(?![0-9])" % re.escape(r), q) for r in rows):
            probs.append("question: shares a component (row) with the gold")
    else:
        if q.count(CHOICE_SENTENCE) != 1:
            probs.append("question: the declared choice list is missing or repeated")
        body = q.replace(CHOICE_SENTENCE, "")
        if re.search(r"(?<![A-Za-z'])[A-D](?![A-Za-z])(?! [a-z])", body):
            probs.append("question: an option letter appears outside the declared choice list")
    return probs


# =================================================================== helpers
def _lum(a):
    a = np.asarray(a, dtype=float)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


@lru_cache(None)
def _font(size, bold=False):
    # Checked explicitly: given a missing path, PIL silently falls back to a same-named host font.
    path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if not path.is_file():
        raise FileNotFoundError(f"label font {path} not found")
    return ImageFont.truetype(str(path), size)


def _text_cover(text, xy, size, bold, box):
    """Ink coverage (0..1) of `text` drawn at absolute xy, cropped to box=(x0,y0,x1,y1).

    The canvas origin is an integer offset left/above xy, so the text keeps the
    same sub-pixel phase as on the page (PIL truncates negative coordinates
    differently, which would shift a label by a pixel)."""
    x0, y0, x1, y1 = box
    ox, oy = int(math.floor(min(x0, xy[0]))) - 2, int(math.floor(min(y0, xy[1]))) - 2
    canvas = Image.new("L", (x1 - ox + 2, y1 - oy + 2), 255)
    ImageDraw.Draw(canvas).text((xy[0] - ox, xy[1] - oy), text, fill=0, font=_font(size, bold))
    a = (255.0 - np.asarray(canvas, dtype=float)) / 255.0
    return a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]


def _text_bbox(text, xy, size, bold):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    return d.textbbox(xy, text, font=_font(size, bold))


def _read_label(arr, xy, size, bold, candidates):
    """Template-match the label drawn at xy among candidates (normalised cross-correlation of ink
    coverage).  Returns (best, score_best, score_second)."""
    boxes = [_text_bbox(t, xy, size, bold) for t in candidates]
    x0 = max(int(math.floor(min(b[0] for b in boxes))) - 1, 0)
    y0 = max(int(math.floor(min(b[1] for b in boxes))) - 1, 0)
    x1 = min(int(math.ceil(max(b[2] for b in boxes))) + 1, arr.shape[1])
    y1 = min(int(math.ceil(max(b[3] for b in boxes))) + 1, arr.shape[0])
    seen = np.clip((255.0 - _lum(arr[y0:y1, x0:x1])) / 225.0, 0, 1)
    scores = []
    for t in candidates:
        m = _text_cover(t, xy, size, bold, (x0, y0, x1, y1))
        a, b = m - m.mean(), seen - seen.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum()))
        scores.append((float((a * b).sum()) / den if den else 0.0, t))
    scores.sort(reverse=True)
    return scores[0][1], scores[0][0], (scores[1][0] if len(scores) > 1 else 0.0)


def _check_label(arr, xy, size, bold, expected, candidates, where, problems, boxes):
    best, s1, s2 = _read_label(arr, xy, size, bold, candidates)
    if best != expected or s1 < 0.8 or (len(candidates) > 1 and s1 - s2 < 0.1):
        problems.append(f"image: label at {where} reads {best!r} (score {s1:.2f} vs {s2:.2f}), expected {expected!r}")
    b = _text_bbox(expected, xy, size, bold)
    boxes.append((b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2))


def _outside_known(arr, boxes, bg=WHITE):
    """Number of pixels outside every allowed box that are not background."""
    allowed = np.zeros(arr.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        allowed[max(int(math.floor(y0)), 0):int(math.ceil(y1)) + 1, max(int(math.floor(x0)), 0):int(math.ceil(x1)) + 1] = True
    stray = np.any(arr != np.array(bg, dtype=arr.dtype), axis=2) & ~allowed
    return int(stray.sum())


# =================================================================== boolean
def _rot_cw(g, turns):
    for _ in range(turns % 4):
        g = [list(row) for row in zip(*g[::-1])]
    return g


def _op(a, b, k):
    return [[TABLES[k][2 * x + y] for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _grid_str(g):
    return "/".join("".join(str(v) for v in row) for row in g)


_STEP_LIST = re.compile(r" Steps: (\d+ degrees, [XYZ], I\d(?:; \d+ degrees, [XYZ], I\d)*)\. ")
_STEP = re.compile(r"(\d+) degrees, ([XYZ]), I(\d)")


def _parse_boolean_question(q):
    lists = _STEP_LIST.findall(q)
    steps = [(int(d), lab, int(inp)) for d, lab, inp in _STEP.findall(lists[0])] if len(lists) == 1 else []
    m = re.search(r"final grid as (\d) rows of (\d) digits", q)
    if not steps or not m or m.group(1) != m.group(2):
        return None
    n = int(m.group(1))
    return dict(steps=steps, n=n, expected=_boolean_question(steps, n))


def _boolean_layout(tier, n, stages):
    """Grid regions (name, x0, y0, cell) and labels (text, xy, size) of the published page layout."""
    regions, labels = [], []
    if tier in ("hard", "xhard"):
        size = 448 if tier == "hard" else 588
        if tier == "hard":
            cell, gap = 18, 24
            tile = n * cell
            left = (size - (3 * tile + 2 * gap)) // 2
            positions = [(left, 8 + k * 84) for k in range(4)]
            input_top = 362
        else:
            cell, gap = 16, 18
            tile = n * cell
            positions = [(9 + (k % 2) * (size // 2), 30 + (k // 2) * 130) for k in range(6)]
            input_top = 460
        for k, (left, top) in enumerate(positions[:2 * stages]):
            op, j = divmod(k, 2)
            for g in range(3):
                regions.append((f"demo-{op}-{j}-{g}", int(left + g * (tile + gap)), int(top), cell))
            labels.append(("XYZ"[op], (left + tile + (gap - 10) // 2, top + tile / 2 - 9), 13, "op"))
            labels.append(("=", (left + 2 * tile + gap + (gap - 10) // 2, top + tile / 2 - 9), 13, "eq"))
        count = stages + 1
        stride = size / count
        for k in range(count):
            left = round(k * stride + (stride - tile) / 2)
            regions.append((f"input-{k}", int(left), input_top, cell))
            labels.append((f"I{k + 1}", (left + tile / 2 - 8, input_top - 23), 13, "input"))
    else:  # xxhard, symbolic.logic layout
        size, cell = 588, 11
        tile = n * cell
        gap, rowh = 18, tile + 42
        for j in range(2 * stages):
            y = 24 + (j // 2) * rowh
            left = 10 + (j % 2) * (size // 2)
            for g in range(3):
                regions.append((f"demo-{j // 2}-{j % 2}-{g}", left + g * (tile + gap), y, cell))
            labels.append(("XYZ"[j // 2], (left + tile + 3, y + tile / 2 - 8), 12, "op"))
            labels.append(("=", (left + 2 * tile + gap + 3, y + tile / 2 - 8), 12, "eq"))
        top = 24 + stages * rowh + 36
        stride = size // 4
        for k in range(stages + 1):
            x = k * stride + (stride - tile) // 2
            regions.append((f"input-{k}", x, top, cell))
            labels.append((f"I{k + 1}", (x + tile / 2 - 8, top - 20), 12, "input"))
    return regions, labels


def _decode_grid(arr, x0, y0, cell, n, patterns=None):
    """Cell value from the 3x3 centre patch; also the inset-interior luminance for a legibility margin.
    If `patterns` is a dict {0: set, 1: set}, the raw bytes of every cell's WHOLE square (cell x cell
    px, its own border included: cells are painted in row-major order, each over its full square) are
    added under the decoded value."""
    grid, centre, interior = [], [], []
    for r in range(n):
        row = []
        for c in range(n):
            cx, cy = x0 + c * cell + cell // 2, y0 + r * cell + cell // 2
            lc = float(_lum(arr[cy - 1:cy + 2, cx - 1:cx + 2]).mean())
            li = float(_lum(arr[y0 + r * cell + 2:y0 + (r + 1) * cell - 1, x0 + c * cell + 2:x0 + (c + 1) * cell - 1]).mean())
            v = int(lc < 180)
            row.append(v)
            centre.append((lc, v))
            interior.append((li, v))
            if patterns is not None:
                patterns[v].add(arr[y0 + r * cell:y0 + (r + 1) * cell, x0 + c * cell:x0 + (c + 1) * cell].tobytes())
        grid.append(row)
    return grid, centre, interior


def _verify_boolean(rec, arr, problems):
    tier, m, q = rec["tier"], rec["meta"], rec["question"]
    fam = "composed_boolean" if tier == "xxhard" else "joint_demo_composed_boolean"
    if m.get("family") != fam:
        problems.append(f"meta: family {m.get('family')!r} is not {fam!r} for {tier}")
        return
    want_n, want_stages = {"hard": (4, 2), "xhard": (5, 3), "xxhard": (7, 3)}[tier]
    parsed = _parse_boolean_question(q)
    if parsed is None:
        problems.append("question: cannot parse the steps / grid size")
        return
    if q != parsed["expected"]:
        problems.append("question: text differs from the declared boolean-grid template")
    steps, n = parsed["steps"], parsed["n"]
    problems += _format_example_problems(q, str(rec.get("answer", "")), "grid")
    stages = len(steps)
    if (n, stages) != (want_n, want_stages):
        problems.append(f"question: {n}x{n} grid with {stages} steps, tier {tier} needs {want_n}x{want_n} with {want_stages}")
        return
    if [s[2] for s in steps] != list(range(2, stages + 2)) or [s[1] for s in steps] != list("XYZ"[:stages]):
        problems.append("question: steps do not use X,Y,Z with I2,I3,I4 in order")
    if steps[0][0] != 0 or any(d not in (90, 180, 270) for d, _, _ in steps[1:]):
        problems.append("question: step rotations outside the declared form")
    turns = [d // 90 for d, _, _ in steps]
    if any(d % 90 for d, _, _ in steps):
        problems.append("question: rotation is not a multiple of 90")
        return
    # ---- meta consistency with the question
    if m.get("n") != n or m.get("rotations") != turns or len(m.get("inputs", [])) != stages + 1 \
            or len(m.get("operators", [])) != stages or len(m.get("demonstrations", [])) != stages \
            or any(len(p) != 2 for p in m["demonstrations"]):
        problems.append("meta: n / rotations / inputs / operators / demonstrations disagree with the question")
        return
    if tier == "xxhard" and len(set(m["operators"])) != 3:
        problems.append("meta: xxhard operators are not 3 distinct operations")
    if tier in ("hard", "xhard") and len(set(m["operators"])) < 2:
        problems.append("meta: fewer than 2 distinct operators")
    # ---- layout: derived here, compared with what the renderer published
    regions, labels = _boolean_layout(tier, n, stages)
    pub = [(r["name"], r["bounds"][0], r["bounds"][1], r["cell"]) for r in m.get("layout", [])]
    if pub != [(a, b, c, d) for a, b, c, d in regions]:
        problems.append("meta: published layout differs from the derived page layout")
    tile_check = all(list(r["bounds"]) == [x, y, x + n * cell, y + n * cell]
                     for r, (_, x, y, cell) in zip(m.get("layout", []), regions))
    if not tile_check:
        problems.append("meta: published region bounds are not n*cell squares")
    # ---- decode every grid from pixels
    decoded, centres, interiors, boxes = {}, [], [], []
    patterns = {0: set(), 1: set()}
    for name, x0, y0, cell in regions:
        if x0 < 0 or y0 < 0 or x0 + n * cell >= arr.shape[1] or y0 + n * cell >= arr.shape[0]:
            problems.append(f"layout: region {name} outside the image")
            return
        g, c, i = _decode_grid(arr, x0, y0, cell, n, patterns)
        decoded[name] = g
        centres += c
        interiors += i
        boxes.append((x0 - 1, y0 - 1, x0 + n * cell + 1, y0 + n * cell + 1))
    # Whole-cell appearance, not just the centre: every cell of one value must look exactly alike
    # (one style, one palette per image), so a cell partly painted in the other state -- a half
    # cell, an ON ring around an OFF centre, a stroke spilling in from a neighbour -- is caught.
    for v in (0, 1):
        if len(patterns[v]) > 1:
            problems.append(f"image: {len(patterns[v])} different-looking cell interiors decode as {v} "
                            "(cells must be painted wholly in one state)")
    if patterns[0] & patterns[1]:
        problems.append("image: an ON cell and an OFF cell have identical interiors")
    pal = m.get("render_palette")
    if not pal or sum(abs(a - b) for a, b in zip(pal[0], pal[1])) < 120 or _lum(np.array(pal[0])) > 140 \
            or _lum(np.array(pal[1])) < 220:
        problems.append(f"meta: palette {pal} is not dark-on / light-off")
    on_l = [l for l, v in interiors if v]
    off_l = [l for l, v in interiors if not v]
    if on_l and off_l and min(off_l) - max(on_l) < 40:
        problems.append(f"image: cell interiors not separable (off min {min(off_l):.0f} vs on max {max(on_l):.0f})")
    ambiguous = sum(1 for l, _ in centres if 150 <= l <= 210)
    if ambiguous:
        problems.append(f"image: {ambiguous} cell centres have mid-grey luminance")
    # ---- labels
    for text, xy, size, kind in labels:
        cands = {"op": list("XYZ"[:stages]), "eq": ["="], "input": [f"I{k + 1}" for k in range(stages + 1)]}[kind]
        _check_label(arr, xy, size, False, text, cands, f"{kind} {text}", problems, boxes)
    stray = _outside_known(arr, boxes)
    if stray:
        problems.append(f"image: {stray} non-white pixels outside the grids and labels")
    # ---- decoded grids must be the meta grids
    want = {f"input-{k}": g for k, g in enumerate(m["inputs"])}
    for op, panels in enumerate(m["demonstrations"]):
        for j, panel in enumerate(panels):
            for g, grid in enumerate(panel):
                want[f"demo-{op}-{j}-{g}"] = grid
    bad = [k for k in want if decoded.get(k) != want[k]]
    if bad:
        problems.append(f"image: decoded grids differ from meta in {len(bad)} regions, e.g. {bad[0]}")
    # ---- solve from the DECODED image + the question, enumerating every operator assignment
    allowed = []
    for op in range(stages):
        panels = [[decoded[f"demo-{op}-{j}-{g}"] for g in range(3)] for j in range(2)]
        allowed.append([k for k in range(4) if all(_op(a, b, k) == c for a, b, c in panels)])
    if any(not a for a in allowed):
        problems.append(f"unique: some operator matches no allowed operation ({allowed})")
        return
    answers = set()
    for ops in itertools.product(*allowed):
        cur = decoded["input-0"]
        for (deg, lab, inp), k in zip(steps, ops):
            cur = _op(_rot_cw(cur, deg // 90), decoded[f"input-{inp - 1}"], k)
        answers.add(_grid_str(cur))
    if len(answers) != 1:
        problems.append(f"unique: {len(answers)} different final grids are consistent with the demonstrations")
    if answers != {rec["answer"]}:
        problems.append("correct: recomputed grid differs from the gold answer")
    if _grid_str(m.get("truth", [])) != rec["answer"]:
        problems.append("meta: truth grid differs from the answer")
    if all(len(a) == 1 for a in allowed) and [a[0] for a in allowed] != m["operators"]:
        problems.append("meta: operators differ from the ones the demonstrations identify")
    if not re.fullmatch(r"[01]{%d}(?:/[01]{%d}){%d}" % (n, n, n - 1), rec["answer"]):
        problems.append("answer: not n rows of n binary digits joined by '/'")
    if rec.get("answer_type") != "text" or rec.get("choices") is not None:
        problems.append("record: boolean tiers must be answer_type 'text' without choices")


# =================================================================== layouts (base page)
def _attr_layout(rows, cols, S=448):
    m_top, m_h = int(S * 0.05), int(S * 0.55)
    box = min(int(S * 0.86) // cols, m_h // rows)
    gx0 = (S - box * cols) // 2
    cells = {(r, c): (gx0 + c * box, m_top + r * box, box) for r in range(rows) for c in range(cols)}
    top = m_top + rows * box + int(S * 0.03)
    return cells, _option_layout(top, S)


def _overlay_layout(S=448):
    m_top, box, gap = int(S * 0.04), int(S * 0.26), int(S * 0.02)
    gx0 = (S - (box * 3 + gap * 2)) // 2
    cells = {(r, c): (gx0 + c * (box + gap), m_top + r * (box + gap), box) for r in range(2) for c in range(3)}
    top = m_top + 2 * (box + gap) + int(S * 0.03)
    return cells, _option_layout(top, S)


def _option_layout(top, S=448):
    pad = int(S * 0.025)
    pw = (S - 2 * pad - 3 * pad) // 4
    ph = S - top - pad - int(S * 0.04)
    inner = top + int(S * 0.05)
    cbox = min(pw - 8, ph - int(S * 0.06))
    out = []
    for i in range(4):
        px = pad + i * (pw + pad)
        out.append(dict(panel=(px, top, px + pw, top + ph), label_xy=(px + 5, top + 3),
                        box=(px + (pw - cbox) // 2, inner, cbox)))
    return out


def _check_frame(arr, cells, options, hidden, q_frac, problems):
    """Option labels, a hidden cell that shows only the '?' mark, and nothing outside the known elements."""
    boxes = [(x - 2, y - 2, x + b + 2, y + b + 2) for (x, y, b) in cells.values()]
    for i, o in enumerate(options):
        x0, y0, x1, y1 = o["panel"]
        boxes.append((x0 - 2, y0 - 2, x1 + 2, y1 + 2))
        _check_label(arr, o["label_xy"], 16, True, "ABCD"[i], list("ABCD"), f"option {'ABCD'[i]}", problems, [])
    # the '?' is bold DejaVu at q_frac*box, centred on its own ink box
    x, y, b = cells[hidden]
    size = int(b * q_frac)
    tb = _text_bbox("?", (0, 0), size, True)
    xy = (x + b // 2 - (tb[2] - tb[0]) // 2, y + b // 2 - (tb[3] - tb[1]) // 2 - tb[1])
    _, score, _ = _read_label(arr, xy, size, True, ["?"])
    if score < 0.8:
        problems.append(f"image: the hidden cell does not show the '?' mark (score {score:.2f})")
    qb = _text_bbox("?", xy, size, True)
    inner = arr[y + 4:y + b - 3, x + 4:x + b - 3]
    Y, X = np.mgrid[y + 4:y + b - 3, x + 4:x + b - 3]
    on_q = (X >= qb[0] - 2) & (X <= qb[2] + 2) & (Y >= qb[1] - 2) & (Y <= qb[3] + 2)
    leak = np.any(inner != np.array(WHITE, dtype=inner.dtype), axis=2) & ~on_q
    spread = inner.astype(int).max(axis=2) - inner.astype(int).min(axis=2)
    if int(leak.sum()) or int((spread > 40).sum()):
        problems.append("image: the hidden cell is not blank apart from the '?' (answer leaked or cell drawn)")
    stray = _outside_known(arr, boxes)
    if stray:
        problems.append(f"image: {stray} non-white pixels outside the matrix and option panels")


# =================================================================== overlay
def _unit_tiles(kind, R, C):
    h = math.sqrt(3) / 2
    polys = []
    for r in range(R):
        for c in range(C):
            if kind == "square":
                polys.append([(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)])
            elif kind == "triangle":
                x, y = c * 0.5, r * h
                polys.append([(x, y + h), (x + 0.5, y), (x + 1, y + h)] if c % 2 == 0 else
                             [(x, y), (x + 1, y), (x + 0.5, y + h)])
            else:  # flat-top hexagons, odd columns shifted down by half a row
                cx, cy = c * 0.75, r * h + (h / 2 if c % 2 else 0)
                polys.append([(cx + 0.5 * math.cos(math.pi * k / 3), cy + 0.5 * math.sin(math.pi * k / 3)) for k in range(6)])
    return polys


TILE_ERODE = 2   # px trimmed off each tile polygon before its whole interior must be one colour


@lru_cache(None)
def _tile_interiors(kind, R, C, box):
    """Per tile, the pixel indices (local to the figure box) of its polygon eroded by TILE_ERODE px.
    The polygon offset inside the box depends only on (kind, R, C, box), so every figure of one
    size shares these masks (an integer box origin shifts the raster exactly)."""
    polys = _unit_tiles(kind, R, C)
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (box - 4) / max(w, h)
    offx, offy = (box - w * scale) / 2, (box - h * scale) / 2
    out = []
    for poly in polys:
        pts = [(offx + (x - min(xs)) * scale, offy + (y - min(ys)) * scale) for x, y in poly]
        mk = Image.new("L", (box + 2, box + 2), 0)
        ImageDraw.Draw(mk).polygon(pts, fill=1)
        out.append(np.nonzero(ndimage.binary_erosion(np.asarray(mk, bool), iterations=TILE_ERODE)))
    return out


@lru_cache(None)
def _tile_background(kind, R, C, box):
    """Pixel indices (local to the figure box) that are neither within 2 px of a tile nor on the
    box frame: margins and the partial notches along a tiling's border, which must stay background."""
    polys = _unit_tiles(kind, R, C)
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (box - 4) / max(w, h)
    offx, offy = (box - w * scale) / 2, (box - h * scale) / 2
    mk = Image.new("L", (box + 2, box + 2), 0)
    d = ImageDraw.Draw(mk)
    for poly in polys:
        d.polygon([(offx + (x - min(xs)) * scale, offy + (y - min(ys)) * scale) for x, y in poly], fill=1, outline=1)
    near = ndimage.binary_dilation(np.asarray(mk, bool), iterations=2)
    region = np.zeros_like(near)
    region[3:box - 2, 3:box - 2] = True
    return np.nonzero(region & ~near)


def _decode_overlay(arr, kind, R, C, x0, y0, box, on, bg=WHITE):
    """Tile states from the centroid and 4 interior points, then the WHOLE eroded tile must be that
    one colour (exactly on-colour or exactly white), so partial paint between samples is caught.
    Everything in the box that is not near a tile must be the background colour bg."""
    polys = _unit_tiles(kind, R, C)
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (box - 4) / max(w, h)
    ox, oy = x0 + (box - w * scale) / 2, y0 + (box - h * scale) / 2
    crop = arr[y0:y0 + box + 2, x0:x0 + box + 2].astype(int)
    on_px = np.all(crop == np.array(on), axis=2)
    white_px = np.all(crop == np.array(WHITE), axis=2)
    interiors = _tile_interiors(kind, R, C, box)
    mask, unclear = [], 0
    for k, poly in enumerate(polys):
        pts = [(ox + (x - min(xs)) * scale, oy + (y - min(ys)) * scale) for x, y in poly]
        gx, gy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        samples = [(gx, gy)] + [(gx + 0.4 * (px - gx), gy + 0.4 * (py - gy)) for px, py in pts]
        states = set()
        for sx, sy in samples:
            px = tuple(int(v) for v in arr[int(round(sy)), int(round(sx))])
            states.add(1 if px == tuple(on) else 0 if px == WHITE else -1)
        inside = interiors[k]
        whole = (on_px if states == {1} else white_px)[inside]
        if len(states) != 1 or -1 in states or not len(whole) or not whole.all():
            unclear += 1
        mask.append(1 if states == {1} else 0)
    back = crop[_tile_background(kind, R, C, box)]
    stray = int(np.any(back != np.array(bg), axis=1).sum())
    return mask, unclear + (1 if stray else 0), scale


def _verify_overlay(rec, arr, problems):
    tier, m = rec["tier"], rec["meta"]
    dims = {"easy": {"square": (4, 4), "triangle": (4, 6), "hexagon": (3, 4)},
            "medium": {"square": (5, 5), "triangle": (5, 8), "hexagon": (4, 5)}}[tier]
    kind = m.get("grid_kind")
    if kind not in dims or (m.get("R"), m.get("C")) != dims[kind]:
        problems.append(f"meta: overlay grid {kind} {m.get('R')}x{m.get('C')} is not the {tier} size")
        return
    R, C = dims[kind]
    on = tuple(m["on_color"])
    if sum(abs(a - b) for a, b in zip(on, WHITE)) < 150:
        problems.append("meta: on colour too close to white")
    cells, options = _overlay_layout()
    decoded, unclear, min_scale = {}, 0, 1e9
    for (r, c), (x, y, b) in cells.items():
        if (r, c) == (1, 2):
            continue
        decoded[(r, c)], u, s = _decode_overlay(arr, kind, R, C, x, y, b, on)
        unclear, min_scale = unclear + u, min(min_scale, s)
    opt_masks = []
    for o in options:
        x, y, b = o["box"]
        mk, u, s = _decode_overlay(arr, kind, R, C, x, y, b, on, bg=(248, 248, 250))
        opt_masks.append(mk)
        unclear, min_scale = unclear + u, min(min_scale, s)
    if unclear:
        problems.append(f"image: {unclear} tiles (or figure backgrounds) are not uniformly on-colour or white")
    if min_scale < 7:
        problems.append(f"image: tiles drawn at {min_scale:.1f} px per unit (< 7)")
    _check_frame(arr, cells, options, (1, 2), 0.42, problems)
    rows = m["row_masks"]
    for (r, c), mk in decoded.items():
        if list(rows[r][c]) != mk:
            problems.append(f"image: decoded figure ({r + 1},{c + 1}) differs from meta")
    for i, mk in enumerate(opt_masks):
        if list(m["options"][i]["mask"]) != mk:
            problems.append(f"image: decoded option {'ABCD'[i]} differs from meta")
    # solve from the decoded image
    a, b, c = decoded[(0, 0)], decoded[(0, 1)], decoded[(0, 2)]
    ops = [k for k in range(4) if [TABLES[k][2 * x + y] for x, y in zip(a, b)] == c]
    preds = {tuple(TABLES[k][2 * x + y] for x, y in zip(decoded[(1, 0)], decoded[(1, 1)])) for k in ops}
    if len(preds) != 1:
        problems.append(f"unique: row 1 leaves {len(ops)} operations predicting {len(preds)} figures")
        return
    pred = next(iter(preds))
    if len({tuple(mk) for mk in opt_masks}) != 4:
        problems.append("unique: two options show the same figure")
    if list(pred) != list(rows[1][2]):
        problems.append("meta: hidden figure differs from the recomputed one")
    hits = [tuple(mk) == pred for mk in opt_masks]
    if sum(hits) != 1:
        problems.append(f"unique: {sum(hits)} options match the recomputed figure")
        return
    if rec["answer"] != "ABCD"[hits.index(True)]:
        problems.append(f"correct: recomputed option {'ABCD'[hits.index(True)]} != gold {rec['answer']}")


# =================================================================== attribute
def _period(shape, count):
    return 360.0 / count if shape == "circle" else 360.0 / math.gcd(SIDES[shape], count)


def _rot_gap(a, b, p):
    d = (a - b) % p
    return min(d, p - d)


def _glyph_key(g):
    p = _period(g["shape"], int(g["count"]))
    r = round(float(g["rot"]) % p, 6)
    return (g["shape"], tuple(int(x) for x in g["color"]), int(g["count"]), round(float(g["size"]), 6),
            0.0 if abs(r - p) < 1e-6 else r)


def _attribute_predictions(grid, rows, cols, n_change):
    """Every program the question allows, per attribute, consistent with the visible cells."""
    hidden = (rows - 1, cols - 1)
    vis = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != hidden]
    g00 = grid[0][0]
    progs = {}
    progs["count"] = [(False, lambda r, c, k=int(g00["count"]): k), (True, lambda r, c, k=int(g00["count"]): k + r + c)]
    for name in ("color", "shape"):
        first = [grid[0][c][name] for c in range(cols)]
        lst = [(False, lambda r, c, x=g00[name]: x)]
        if cols == 3 and len({str(x) for x in first}) == 3:
            lst += [(True, lambda r, c, cyc=first, s=s: cyc[(c + s * r) % 3]) for s in (1, 2)]
        progs[name] = lst
    progs["size"] = [(False, lambda r, c, x=float(g00["size"]): x)] + \
        [(True, lambda r, c, k=k, s=s: SIZES[(k + c + s * r) % 3]) for k in range(3) for s in (1, 2)]
    p00 = _period(g00["shape"], int(g00["count"]))
    bases = [(float(g00["rot"]) + j * p00) % 360 for j in range(round(360 / p00))]
    progs["rot"] = [(False, lambda r, c, b=b: b) for b in bases] + \
        [(True, lambda r, c, b=b, st=st: (b + ROW_ROT * r + st * c) % 360) for b in bases for st in ROT_STEPS]
    ok = {}
    for name, lst in progs.items():
        keep = []
        for changing, fn in lst:
            good = True
            for r, c in vis:
                obs, v = grid[r][c], fn(r, c)
                if name == "rot":
                    good = _rot_gap(v, float(obs["rot"]), _period(obs["shape"], int(obs["count"]))) < 1e-6
                elif name == "size":
                    good = abs(v - float(obs["size"])) < 1e-6
                elif name == "color":
                    good = tuple(v) == tuple(obs["color"])
                else:
                    good = v == obs[name]
                if not good:
                    break
            if good:
                keep.append((changing, fn(*hidden)))
        ok[name] = keep
    names = ("count", "color", "shape", "size", "rot")
    preds, n_prog = {}, 0
    for combo in itertools.product(*(ok[k] for k in names)):
        if sum(ch for ch, _ in combo) != n_change:
            continue
        g = dict(zip(names, (v for _, v in combo)))
        if g["count"] < 1:
            continue
        n_prog += 1
        preds[_glyph_key(g)] = g
    return preds, n_prog


def _glyph_points(shape, cx, cy, R, rot):
    a0 = math.radians(rot) - math.pi / 2
    if shape == "star":
        return [(cx + (R if k % 2 == 0 else 0.45 * R) * math.cos(a0 + math.pi * k / 5),
                 cy + (R if k % 2 == 0 else 0.45 * R) * math.sin(a0 + math.pi * k / 5)) for k in range(10)]
    n = SIDES[shape]
    return [(cx + R * math.cos(a0 + 2 * math.pi * k / n), cy + R * math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)]


def _pip_centres(cx, cy, R, rot, count):
    a0 = math.radians(rot) - math.pi / 2 + 0.6
    return [(cx + 0.42 * R * math.cos(a0 + 2 * math.pi * k / count), cy + 0.42 * R * math.sin(a0 + 2 * math.pi * k / count))
            for k in range(count)]


def _silhouette(shape, cx, cy, R, rot, count, X, Y):
    from matplotlib.path import Path as MPath
    if shape == "circle":
        inside = (X - cx) ** 2 + (Y - cy) ** 2 <= (R + 1) ** 2
    else:
        pts = np.stack([X.ravel(), Y.ravel()], 1)
        inside = MPath(_glyph_points(shape, cx, cy, R, rot)).contains_points(pts).reshape(X.shape)
    dot = max(3, int(R * 0.13)) + 1
    for px, py in _pip_centres(cx, cy, R, rot, count):
        inside |= (X - px) ** 2 + (Y - py) ** 2 <= dot ** 2
    return inside


def _stray_in_box(arr, box, bg, cx, cy, R0, g, exclude=()):
    """Non-background pixels inside box that are neither on the expected glyph (within 3 px)
    nor inside an excluded rectangle (the option letter)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    sub = arr[y0:y1, x0:x1].astype(int)
    Y, X = np.mgrid[y0:y1, x0:x1].astype(float)
    glyph = _silhouette(g["shape"], cx, cy, R0 * float(g["size"]), float(g["rot"]), int(g["count"]), X, Y)
    allowed = ndimage.distance_transform_edt(~glyph) <= 3.0   # drawn outline lies within sqrt(5) px
    if g["shape"] != "circle":  # rasterised outlines overshoot the pixel-centre mask at acute vertices
        for vx, vy in _glyph_points(g["shape"], cx, cy, R0 * float(g["size"]), float(g["rot"])):
            allowed |= (X - vx) ** 2 + (Y - vy) ** 2 <= 9
    for ex0, ey0, ex1, ey1 in exclude:
        allowed |= (X >= ex0) & (X <= ex1) & (Y >= ey0) & (Y <= ey1)
    ink = np.abs(sub - np.array(bg)).sum(2) > 30
    return int((ink & ~allowed).sum())


def _decode_glyph(arr, cx, cy, R0, g, bg):
    """Decode one glyph and compare with meta glyph g.  Returns a list of mismatching attributes."""
    bad = []
    size, shape, count, rot = float(g["size"]), g["shape"], int(g["count"]), float(g["rot"])
    R = R0 * size
    span = int(R0 * 1.15) + 5
    x0, y0 = int(cx) - span, int(cy) - span
    crop = arr[y0:y0 + 2 * span, x0:x0 + 2 * span].astype(int)
    Y, X = np.mgrid[y0:y0 + 2 * span, x0:x0 + 2 * span].astype(float)
    # colour: the pip-free core disc must be one flat colour equal to meta
    core = (X - cx) ** 2 + (Y - cy) ** 2 <= (0.22 * R) ** 2
    vals, counts = np.unique(crop[core].reshape(-1, 3), axis=0, return_counts=True)
    top = tuple(int(v) for v in vals[counts.argmax()])
    if counts.max() < 0.9 * core.sum() or top != tuple(int(v) for v in g["color"]):
        bad.append(f"colour {top}")
    # silhouette: everything that is not background, holes (pips) filled
    sil = ndimage.binary_fill_holes(np.abs(crop - np.array(bg)).sum(2) > 30)
    lab, nlab = ndimage.label(sil)
    if nlab > 1:  # keep the component under the centre
        sil = lab == lab[int(cy) - y0, int(cx) - x0]

    def iou(sh, sz, rt):
        m = _silhouette(sh, cx, cy, R0 * sz, rt, count, X, Y)
        return np.logical_and(m, sil).sum() / max(np.logical_or(m, sil).sum(), 1)
    true_iou = iou(shape, size, rot)
    if true_iou < 0.85:
        bad.append(f"outline iou {true_iou:.2f}")
    # shape and size are decoded JOINTLY: e.g. a circle at 0.85 and a hexagon at 1.0 have
    # similar silhouettes, so the meta pair must beat all 17 other (shape, size) pairs.
    for sh in SHAPES:
        for sz in SIZES:
            if (sh, round(sz, 6)) != (shape, round(size, 6)) and iou(sh, sz, rot) >= true_iou - 0.01:
                bad.append(f"outline reads as {sh} size {sz}")
    p = _period(shape, count)
    for j in range(1, count):
        alt = rot + j * 360.0 / count
        if _rot_gap(alt, rot, p) > 1e-6 and iou(shape, size, alt) >= true_iou - 0.005:
            bad.append(f"rotation reads as {alt % 360:.0f}")
    # pips: white blobs enclosed by the glyph, count and ring angles
    white = (crop.min(2) > 235) & sil
    lab, nb = ndimage.label(white)
    keep = [i for i in range(1, nb + 1) if (lab == i).sum() >= 2]
    if len(keep) != count:
        bad.append(f"{len(keep)} pips")
    else:
        cents = ndimage.center_of_mass(white, lab, keep)
        want = _pip_centres(cx, cy, R, rot, count)
        step = 360.0 / count
        for (py, px) in cents:
            ang = math.degrees(math.atan2(py + y0 - cy, px + x0 - cx))
            err = min(_rot_gap(ang, math.degrees(math.atan2(wy - cy, wx - cx)), 360) for wx, wy in want)
            dist = math.hypot(px + x0 - cx, py + y0 - cy)
            # centroid of a ~2-4 px pip interior is good to ~1.5 px: allow that much angle at this radius
            tol = max(6.0, math.degrees(math.atan2(1.5, max(dist, 1.0))) + 1.5)
            if err > tol or abs(dist - 0.42 * R) > 0.15 * R + 1.5:
                bad.append(f"pip at {ang:.0f} deg / r {dist:.1f} off the ring (step {step:.0f})")
                break
    return bad


def _distinguishing(gold, opt):
    """Attributes in which a distractor visibly differs from the gold glyph."""
    out = []
    if opt["shape"] != gold["shape"]:
        out.append("shape")
    if int(opt["count"]) != int(gold["count"]):
        out.append("count")
    if sum(abs(int(a) - int(b)) for a, b in zip(opt["color"], gold["color"])) >= COLOR_MIN:
        out.append("color")
    if not out and _rot_gap(float(opt["rot"]), float(gold["rot"]), _period(gold["shape"], int(gold["count"]))) >= ROT_MIN:
        out.append("rotation")
    return out


def _verify_attribute(rec, arr, problems):
    tier, m, q = rec["tier"], rec["meta"], rec["question"]
    rows, cols = m.get("rows"), m.get("cols")
    want_rows, want_n = (2, 3) if tier == "easy" else (3, 4)
    mm = re.search(r"Exactly (\d) of five attributes", q)
    if not mm or q != ATTR_TEMPLATE.format(n=mm.group(1)):
        problems.append("question: text differs from the declared attribute-matrix template")
        return
    problems += _format_example_problems(q, str(rec.get("answer", "")), "choice")
    n_change = int(mm.group(1))
    if (rows, cols, n_change, m.get("changing_attributes")) != (want_rows, 3, want_n, want_n):
        problems.append(f"meta/question: {rows}x{cols} matrix with {n_change} changing, tier {tier} needs "
                        f"{want_rows}x3 with {want_n}")
        return
    grid = m["grid_attr"]
    opts = [o["attr"] for o in m["options"]]
    for g in [x for row in grid for x in row] + opts:
        if g["shape"] not in SHAPES or not 1 <= int(g["count"]) <= 12 or \
                not any(abs(float(g["size"]) - s) < 1e-6 for s in SIZES):
            problems.append(f"meta: glyph outside the declared value sets {g}")
            return
    # ---- image decode.  ONE glyph scale for the page (the question says options are drawn at the
    # matrix scale): unit radius 0.34 x min(matrix cell, option box).  Decoding every glyph at that
    # radius, with size decoded jointly with shape, is what proves the drawn sizes are comparable.
    cells, options = _attr_layout(rows, cols)
    r0 = 0.34 * min(min(b for _, _, b in cells.values()), min(o["box"][2] for o in options))
    if abs(float(m.get("glyph_unit_radius", -1)) - r0) > 1e-4:
        problems.append(f"meta: glyph_unit_radius {m.get('glyph_unit_radius')} is not the page's {r0:.4f}")
    for (r, c), (x, y, b) in cells.items():
        if (r, c) == (rows - 1, cols - 1):
            continue
        bad = _decode_glyph(arr, x + b / 2, y + b / 2, r0, grid[r][c], WHITE)
        stray = _stray_in_box(arr, (x + 3, y + 3, x + b - 2, y + b - 2), WHITE, x + b / 2, y + b / 2, r0, grid[r][c])
        if stray:
            bad.append(f"{stray} stray pixels in the cell")
        if bad:
            problems.append(f"image: cell ({r + 1},{c + 1}) decodes differently from meta: {', '.join(bad[:3])}")
    for i, o in enumerate(options):
        x, y, b = o["box"]
        bad = _decode_glyph(arr, x + b / 2, y + b / 2, r0, opts[i], (248, 248, 250))
        px0, py0, px1, py1 = o["panel"]
        lb = _text_bbox("ABCD"[i], o["label_xy"], 16, True)
        stray = _stray_in_box(arr, (px0 + 3, py0 + 3, px1 - 2, py1 - 2), (248, 248, 250), x + b / 2, y + b / 2,
                              r0, opts[i], exclude=[(lb[0] - 2, lb[1] - 2, lb[2] + 2, lb[3] + 2)])
        if stray:
            bad.append(f"{stray} stray pixels in the panel")
        if bad:
            problems.append(f"image: option {'ABCD'[i]} decodes differently from meta: {', '.join(bad[:3])}")
    _check_frame(arr, cells, options, (rows - 1, cols - 1), 0.45, problems)
    # ---- enumerate the declared rule family
    preds, n_prog = _attribute_predictions(grid, rows, cols, n_change)
    if len(preds) != 1:
        problems.append(f"unique: {n_prog} admissible programs predict {len(preds)} different glyphs")
        return
    key, gold = next(iter(preds.items()))
    if _glyph_key(grid[-1][-1]) != key:
        problems.append("meta: hidden glyph differs from the recomputed one")
    if len({_glyph_key(o) for o in opts}) != 4:
        problems.append("unique: two options show the same glyph")
    hits = [_glyph_key(o) == key for o in opts]
    if sum(hits) != 1:
        problems.append(f"unique: {sum(hits)} options match the recomputed glyph")
        return
    letter = "ABCD"[hits.index(True)]
    if rec["answer"] != letter:
        problems.append(f"correct: recomputed option {letter} != gold {rec['answer']}")
    for i, o in enumerate(opts):
        if not hits[i] and not _distinguishing(gold, o):
            diff = [k for k in ("size", "rot") if _glyph_key(o) != _glyph_key({**o, k: gold[k]})]
            problems.append(f"perceptual: option {'ABCD'[i]} differs from the gold only in {'+'.join(diff) or 'nothing'}"
                            " (size alone is never accepted as the visible difference; rotation gap < 15 deg)")


# =================================================================== entry
def verify(record: dict, image) -> list[str]:
    problems: list[str] = []
    tier = record.get("tier")
    if tier not in TIERS:
        return [f"record: unknown tier {tier!r}"]
    arr = np.asarray(image.convert("RGB"))
    if arr.shape[:2] != (IMAGE_SIZE[tier],) * 2:
        return [f"image: size {arr.shape[1]}x{arr.shape[0]} is not {IMAGE_SIZE[tier]} square for {tier}"]
    m = record.get("meta") or {}
    fam = m.get("family")
    try:
        if tier in ("easy", "medium"):
            if record.get("answer_type") != "choice" or record.get("choices") != list("ABCD") \
                    or record.get("answer") not in list("ABCD"):
                problems.append("record: multiple-choice tiers need answer_type 'choice', choices A-D")
            if len(m.get("options", [])) != 4 or [o.get("label") for o in m["options"]] != list("ABCD"):
                problems.append("meta: need exactly 4 options labelled A-D")
                return problems
            if fam == "explicit_attribute":
                _verify_attribute(record, arr, problems)
            elif fam == "explicit_overlay":
                if record["question"] != OVERLAY_TEMPLATE:
                    problems.append("question: text differs from the declared overlay template")
                problems += _format_example_problems(record["question"], str(record.get("answer", "")), "choice")
                _verify_overlay(record, arr, problems)
            else:
                problems.append(f"meta: unknown family {fam!r} for {tier}")
        else:
            _verify_boolean(record, arr, problems)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        problems.append(f"record: malformed meta ({type(exc).__name__}: {exc})")
    except OSError as exc:   # the label templates need the repository's DejaVu fonts
        problems.append(f"environment: cannot load the label fonts from {FONT_DIR} ({exc}); the verifier needs "
                        "the repository's assets/fonts/DejaVuSans*.ttf")
    return problems
