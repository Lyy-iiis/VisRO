"""logic_matrix -- complete a matrix from attribute, overlay or boolean rules.

Generator routes (shared/puzzles plus generator/):

  easy, medium  generator/logic_low.py -> generator/matrix_base.py
                (explicit_attribute on even index, explicit_overlay on odd index;
                 the checker check_logic_low is the generator's own gate)
  hard, xhard   boolean_grids.py make()/render() + logic_variant.restyle
                (boolean_checks.check is the generator's own gate)
  xxhard        builders.build -> symbolic.logic() + logic_variant.restyle_symbolic_logic
                (oracles.check is the generator's own gate)

What this adapter adds:

* fonts: the renderers are pointed at the repo-root assets/fonts copies, so no
  host font is ever picked up.
* attribute glyph scale: every glyph is drawn at ONE unit radius per image,
  0.34 x min(matrix cell, option box) (easy 30.6 px, medium 27.88 px), and the
  question says so.
* questions: the generator's gates run on the generator text; the release text
  is then rewritten from the item's own parameters:
    attribute  + "Size is the outline's circumradius ... same scale" sentence
    overlay    + the whole-tiles sentence and the attribute return sentence
    boolean    ONE template for hard/xhard/xxhard.
  Every question is at most 240 tokens (LLaVA vicuna + BOS, PaliGemma), worst
  case over all parameters.
* meta is normalised for an independent checker: boolean demonstrations are
  [label][panel][left, right, result] in every tier, the renderer's grid layout
  is published as meta.layout, and the style/palette are recorded for every
  boolean item.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import math

import numpy as np

from visro import api
from visro.seeding import digest

TASK = "logic_matrix"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

FONT_DIR = (_ROOT / 'assets/fonts')
IMAGE_SIZE = {"easy": 448, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}
GENERATOR = {"easy": "generator/logic_low.py:build", "medium": "generator/logic_low.py:build",
             "hard": "generator/boolean_grids.py:make", "xhard": "generator/boolean_grids.py:make",
             "xxhard": "shared/puzzles/symbolic.py:logic"}

TIER_DEFINITIONS = {
    "easy": ("448 px, multiple choice A-D. Even index: explicit_attribute 2x3 glyph matrix, bottom-right hidden, "
             "exactly 3 of 5 attributes change (pip count +1 per column and per row from a 1-3 start; colour and "
             "outline shape each a cycle of 3 distinct values with row shift 1|2; size 0.85/1.0/1.15 cycle with row "
             "shift 1|2; rotation +30|45|60|90 deg per column and +40 deg per row), 3 distractors each flipping 2 "
             "changing attributes (balanced codes 1,2,3); matrix and option glyphs drawn at one scale "
             "(size 1.0 = 30.6 px circumradius). Odd index: explicit_overlay, 2 rows x 3 figures on square 4x4 | "
             "triangle 4x6 | hexagon 3x4 tiles, density 0.42, op OR|AND|XOR|left-minus-right identified by row 1, "
             "3 distractors = gold with 1-2 equal-area tile swaps. Release gate: every distractor differs from the "
             "gold in shape, pip count, colour or a >=15 deg rotation (never only in size)."),
    "medium": ("448 px, multiple choice A-D. Even index: explicit_attribute 3x3 glyph matrix, exactly 4 of 5 "
               "attributes change (same rule family as easy), 3 distractors each flipping 2-3 changing attributes; "
               "matrix and option glyphs drawn at one scale (size 1.0 = 27.88 px circumradius). Odd index: "
               "explicit_overlay on square 5x5 | triangle 5x8 | hexagon 4x5 tiles (otherwise as easy). Same release "
               "gate as easy."),
    "hard": ("448 px, free-form 4x4 binary grid (4 rows of 4 digits joined by '/'). boolean_grids "
             "joint_demo_composed_boolean: 2 stages (rotate 0, then 90|180|270 deg cw), operators X,Y from "
             "OR|AND|XOR|left-minus-right with >=2 distinct, inputs I1-I3 30-70% filled, 2 demonstrations per "
             "operator each leaving exactly 2 candidate ops and jointly 1; gates: truth 25-75% filled, input "
             "dependency >=6, stage effect >=4, wrong-operator effect >=2, wrong-rotation effect >=2, last-input "
             "shortcut <=6 cells; 18 px cells, style index%7, palette (index//7)%5."),
    "xhard": ("588 px, free-form 5x5 binary grid. boolean_grids joint_demo_composed_boolean: 3 stages (rotate 0, then "
              "two of 90|180|270 deg cw), operators X,Y,Z with >=2 distinct, inputs I1-I4 30-70% filled, same "
              "demonstration design; gates: truth 25-75% filled, input dependency >=9, stage effect >=5, "
              "wrong-operator effect >=3, wrong-rotation effect >=3, last-input shortcut <=10 cells; 16 px cells, "
              "style index%7, palette (index//7)%5."),
    "xxhard": ("588 px, free-form 7x7 binary grid. symbolic composed_boolean: 3 stages (rotate 0, then two of "
               "90|180|270 deg cw), operators X,Y,Z = 3 distinct of OR|AND|XOR|left-minus-right, inputs I1-I4 "
               "uniform random bits, 2 demonstrations per operator each alone separating all 4 ops; no "
               "significance gates; 11 px cells, style index%7, palette (index//7)%5. Harder than xhard only in "
               "grid size and cell size, not in operator inference (a single demonstration identifies each "
               "operator; no degeneracy gates)."),
}

_M = None


def _load():
    global _M
    if _M is not None:
        return _M
    from shared.puzzles import builders, drawing, oracles, symbolic
    from tasks.planar_transformations.logic_matrix.generator import (boolean_checks, boolean_grids, logic_low,
                                                                     logic_variant, matrix_base)
    regular, bold = (_ROOT / 'assets/fonts/DejaVuSans.ttf'), (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')
    if not (regular.is_file() and bold.is_file()):
        raise RuntimeError(f"fonts missing under {FONT_DIR}")
    _M = dict(logic_low=logic_low, logic_variant=logic_variant, drawing=drawing, boolean_grids=boolean_grids,
              boolean_checks=boolean_checks, oracles=oracles, builders=builders, symbolic=symbolic,
              matrix_base=matrix_base)
    return _M


# --------------------------------------------------------------------- source
def glyph_unit_radius(rows: int, size: int = 448) -> float:
    """One glyph scale for the whole attribute page: 0.34 x the smaller of the matrix cell and the
    option box of matrix_base's layout (render_problem_image / _render_options).  easy 30.6, medium 27.88."""
    m_top, m_h = int(size * 0.05), int(size * 0.55)
    box = min(int(size * 0.86) // 3, m_h // rows)
    top = m_top + rows * box + int(size * 0.03)
    pad = int(size * 0.025)
    panel_w = (size - 2 * pad - 3 * pad) // 4
    panel_h = size - top - pad - int(size * 0.04)
    cbox = min(panel_w - 8, panel_h - int(size * 0.06))
    return 0.34 * min(box, cbox)


def _render_low_same_scale(M, rec):
    """logic_low.render with every attribute glyph (matrix cells and options) drawn at one unit
    radius.  base draws each glyph at 0.34 x its own box, which put the options at 0.73x (easy) /
    1.10x (medium) the matrix scale.  Only render_attr_cell is swapped, for this call only."""
    src = M["matrix_base"]
    if rec["meta"]["family"] != "explicit_attribute":
        return M["logic_low"].render(rec)
    r0 = glyph_unit_radius(rec["meta"]["rows"])
    original = src.render_attr_cell

    def same_scale(img, d, attr, x0, y0, box):
        src.draw_glyph(d, x0 + box / 2, y0 + box / 2, r0, attr)
    src.render_attr_cell = same_scale
    try:
        return M["logic_low"].render(rec)
    finally:
        src.render_attr_cell = original


def raw(tier: str, seed: int, index: int):
    """The build + render route for one item: (source record, question image).
    Raises on generator gates.  easy/medium draw the attribute glyphs at one scale."""
    M = _load()
    if tier in ("easy", "medium"):
        rec = M["logic_low"].build(tier, seed, index)       # runs check_logic_low internally
        im = _render_low_same_scale(M, rec)
        rec["source_version"] = "low_tiers"
    elif tier in ("hard", "xhard"):
        rec = M["boolean_grids"].make(tier, seed, index)
        M["boolean_checks"].check(rec)
        im = M["logic_variant"].restyle(rec, M["boolean_grids"].render(rec))
        rec["source_version"] = "boolean_grids"
    elif tier == "xxhard":
        rec, im0 = M["builders"].build("logic_matrix", tier, seed, index, M["symbolic"].logic)
        M["oracles"].check(rec)
        regions = M["logic_variant"].symbolic_logic_regions(rec, im0)
        if regions is None:
            raise api.Reject("symbolic layout self-check failed")
        im = M["logic_variant"].restyle_symbolic_logic(rec, im0)
        # symbolic.logic names demo panels flat (demo-<panel>-<grid>, 2 panels per operator);
        # use the boolean_grids scheme demo-<operator>-<panel>-<grid> in every tier.
        for r in regions:
            if r["name"].startswith("demo-"):
                j, g = map(int, r["name"].split("-")[1:])
                r["name"] = f"demo-{j // 2}-{j % 2}-{g}"
        rec["render_regions"] = regions
        rec["source_version"] = "symbolic"
    else:
        raise ValueError(f"unknown tier {tier!r}")
    return rec, im


# ---------------------------------------------------------------- questions
# The generator's own gates check the SOURCE text (check_logic_low compares it with
# attribute_rules.py).  The release text is rebuilt below from the item's parameters,
# after the source text has been matched against its known form, so a drift in the
# fixed code is a Reject rather than a silently different question.
# the attribute and boolean texts are compressed under 240 tokens (LLaVA vicuna
# + BOS and PaliGemma, worst case over every parameter value), stating the same rules.
_ATTR_SOURCE = (     # attribute_rules.py ATTR_RULES, the text check_logic_low gates
    "Use ONLY the following rule family. Exactly {n} of the five attributes (pip count, color, outline shape, size, "
    "rotation) change; every other attribute is constant throughout the matrix. For a changing pip count, moving one "
    "column right adds one pip and moving one row down adds one pip. A changing color or outline shape repeats a fixed "
    "cycle of THREE DISTINCT values from left to right; every next row starts one or two positions later in that same "
    "cycle, using the same row shift throughout. A changing size repeats small, medium, large in that order from left "
    "to right; every next row starts one or two positions later, using the same row shift throughout. A changing "
    "rotation adds a fixed 30, 45, 60, or 90 degrees clockwise per column; every next row starts 40 degrees clockwise "
    "from the previous row start. Rotation moves the outline and its pip ring together. Complete the missing "
    "bottom-right cell. Return the matching option letter A, B, C, or D.")
_OVERLAY_SRC_END = "Return A, B, C, or D."

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

OVERLAY_QUESTION = (
    'Filled tiles are 1, empty 0. In both rows the third figure combines the first two by the '
    'same tile-wise operation: OR, AND, XOR or left-minus-right (filled left, empty right). Infer'
    ' it from the first row; which option completes the second row? Only whole tiles count. '
    + CHOICE_SENTENCE
)


def attribute_question(src_q: str, n: int) -> str:
    if src_q != _ATTR_SOURCE.format(n=n):
        raise api.Reject("unexpected attribute question text")
    return ATTR_TEMPLATE.format(n=n)


def overlay_question(src_q: str) -> str:
    if not src_q.endswith(_OVERLAY_SRC_END) or src_q.count(_OVERLAY_SRC_END) != 1:
        raise api.Reject("unexpected overlay question text")
    return OVERLAY_QUESTION


def _boolean_source_question(n, rotations):
    """boolean_grids.py make(): the hard/xhard source text."""
    q = ("Filled cells represent 1 and light cells 0. Each operator " + ", ".join("XYZ"[:len(rotations)]) +
         " is fixed across its two demonstrations. Each is one of pixelwise OR, AND, XOR, or left-minus-right: a "
         "cell is filled exactly when it is filled on the left and empty on the right. Different operator labels "
         "may represent the same operation. Infer every operator using BOTH of its demonstrations. The grids "
         "labeled I1, I2, I3, I4 as needed are the inputs. Start with I1. ")
    for i, turn in enumerate(rotations):
        q += f"Step {i + 1}: rotate the current grid {90 * turn} degrees clockwise, then apply {'XYZ'[i]} with I{i + 2} as the right operand. "
    return q + (f"Reconstruct the complete final {n} by {n} grid. Return exactly {n} rows of {n} binary digits "
                "separated by /, top to bottom and left to right; 1 means filled and 0 means empty.")


def _symbolic_source_question(n, rotations):
    """shared/puzzles/symbolic.py logic(): the xxhard source text (rank 2, no options)."""
    q = ("Each operator X, Y or Z used in the image is fixed across its two demonstrations. Every operator is one "
         "of pixelwise OR, AND, XOR, or left-minus-right (a cell is filled iff it is filled on the left and empty "
         "on the right). Infer each operator from its demonstrations. The input tiles below the demonstrations "
         "are labeled I1, I2, I3, I4 as needed; any answer options form the bottom row labeled A–D. Start with "
         "input I1. ")
    for k, turn in enumerate(rotations):
        q += f"Step {k + 1}: rotate the current tile {turn * 90} degrees clockwise, then apply operator {'XYZ'[k]} with input I{k + 2} as the right operand. "
    return q + (f"Reconstruct the final {n} by {n} tile. Return {n} rows of {n} binary digits, separated by /; "
                "1 means filled and 0 empty, with rows top-to-bottom and columns left-to-right.")


def boolean_question(n: int, rotations) -> str:
    """The one release template of hard, xhard and xxhard."""
    labels = "XYZ"[:len(rotations)]
    named = " and ".join(labels) if len(labels) == 2 else ", ".join(labels[:-1]) + " and " + labels[-1]
    steps = "; ".join(f"{90 * turn} degrees, {labels[i]}, I{i + 2}" for i, turn in enumerate(rotations))
    return ("Each label (" + named + ") is one tile-wise operation, the same in both its demonstrations (left, "
            "label, right = result): OR, AND, XOR or left-minus-right; labels may share one. Start from I1; each step "
            "turns the current grid clockwise by its angle, then applies its label with the current grid on the left "
            f"and its input on the right. Steps: {steps}. Answer the final grid as {n} rows of {n} digits (1 filled, "
            "0 empty), top to bottom, separated by /.")


# ----------------------------------------------------------------- helpers
def _grid_str(grid) -> str:
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


_SYMMETRY = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "star": 5}


def _period(g) -> float:
    """Rotation period of a glyph: outline symmetry combined with the pip ring."""
    k = int(g["count"])
    if g["shape"] == "circle":
        return 360 / k
    return 360 / math.gcd(_SYMMETRY[g["shape"]], k)


def _glyph_core(g):
    return [g["shape"], [int(x) for x in g["color"]], int(g["count"]), round(float(g["size"]), 6),
            round(float(g["rot"]) % _period(g), 6)]


def core_key_of(meta: dict) -> str:
    """Content-canonical key: what a solver sees, with the answer cell blanked.

    attribute  visible glyphs with rotation reduced modulo the glyph's own
               rotational symmetry (outline x pip ring), hidden cell blanked
    overlay    tessellation kind + the five visible masks
    boolean    n, operators, rotations, inputs (style, palette and the random
               demonstration panels are presentation, not content)
    Matches the core_key the driver deduplicates on.
    """
    fam = meta["family"]
    if fam == "explicit_attribute":
        grid = [[_glyph_core(g) for g in row] for row in meta["grid_attr"]]
        grid[-1][-1] = None
        core = dict(family=fam, grid=grid)
    elif fam == "explicit_overlay":
        rows = [[list(m) for m in row] for row in meta["row_masks"]]
        rows[-1][-1] = None
        core = dict(family=fam, kind=meta["grid_kind"], rows=rows)
    else:
        core = {k: meta[k] for k in ("n", "operators", "rotations", "inputs")}
    return digest(dict(task=TASK, core=core))


# ------------------------------------------------------------------- build
def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in api.TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    rec, im = raw(tier, seed, index)
    if im.mode != "RGB" or im.size != (IMAGE_SIZE[tier],) * 2:
        raise api.Reject(f"unexpected image {im.mode} {im.size}")
    m = rec["meta"]
    meta = {"family": m["family"], "source_version": rec["source_version"],
            "source_generation": {k: v for k, v in rec.get("generation", {}).items() if k != "seed"},
            "image_size": IMAGE_SIZE[tier], "generator": GENERATOR[tier]}
    if tier in ("easy", "medium"):
        meta.update({k: m[k] for k in m if k not in ("family",)})
        meta["options"] = [dict(label="ABCD"[i], attr=o["attr"]) if "attr" in o else dict(label="ABCD"[i], mask=o["mask"])
                           for i, o in enumerate(m["options"])]
        if m["family"] == "explicit_attribute":
            meta["hidden_cell"] = [m["rows"] - 1, m["cols"] - 1]
            meta["glyph_unit_radius"] = round(glyph_unit_radius(m["rows"]), 6)
            question = attribute_question(rec["question"], m["changing_attributes"])
        else:
            meta["hidden_cell"] = [1, 2]
            question = overlay_question(rec["question"])
        answer, answer_type, choices = rec["answer"], "choice", list("ABCD")
    else:
        M = _load()
        lv = M["logic_variant"]
        style = lv.style_for(rec)
        pal_name, on, off = lv.palette_for(rec)
        demos = m["demonstrations"]
        source_q = (_symbolic_source_question if m["family"] == "composed_boolean" else _boolean_source_question)(
            m["n"], m["rotations"])
        if rec["question"] != source_q:
            raise api.Reject("unexpected boolean source question text")
        question = boolean_question(m["n"], m["rotations"])
        if m["family"] == "composed_boolean":          # symbolic.logic stores 2 panels per label, flat
            demos = [demos[2 * i:2 * i + 2] for i in range(len(m["operators"]))]
            sig = M["boolean_grids"].significance([np.asarray(a, dtype=np.uint8) for a in m["inputs"]],
                                           m["operators"], m["rotations"])
        else:
            sig = m["significance"]
        meta.update(n=m["n"], operators=m["operators"], rotations=m["rotations"], demonstrations=demos,
                    inputs=m["inputs"], truth=m["truth"], significance=sig,
                    render_style=style, render_palette=[list(on), list(off)], render_palette_name=pal_name,
                    layout=[dict(name=r["name"], bounds=list(r["bounds"]), cell=r["cell"])
                            for r in rec["render_regions"]])
        if "requirements" in m:
            meta["requirements"] = m["requirements"]
        answer = rec["answer"]
        if answer != _grid_str(m["truth"]):
            raise api.Reject("answer does not spell the truth grid")
        answer_type, choices = "text", None
    meta["image_labels"] = rec.get("image_labels", [])
    return api.Item(question=question, answer=answer, image=im, meta=meta, core_key=core_key_of(meta),
                    answer_type=answer_type, choices=choices)
