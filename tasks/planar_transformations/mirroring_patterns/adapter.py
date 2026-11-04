"""mirroring_patterns: partial (rectangle-restricted) reflections of a colour grid.

Generator:

    easy, medium         generator/mirrors_low.py  build()/panel()
    hard, xhard, xxhard  shared/puzzles/symbolic.py mirrors() via shared/puzzles/builders.py build()

The generator samples the grid, the reflection rectangles and the queried cells
and draws the question image; this adapter keeps all of that and adds:

  * the question states the coordinate convention -- "(row,column)", both
    1-based, rows numbered top to bottom, columns left to right, ranges
    inclusive;
  * white is coded "W" (a "." could be stripped as sentence punctuation);
  * hard/xhard/xxhard reject an item whose answer equals reading the queried
    cells off the INITIAL grid, the gate easy/medium already have
    (in mirrors_low.py);
  * labels are drawn with the committed font assets/fonts/DejaVuSans.ttf (sha
    pinned below), never a host font;
  * core_key covers only the ANSWER-RELEVANT content -- operations, queried
    cells and the colour each one ends up with -- canonicalised over the 8
    whole-grid rotations/mirrors, so two items with the same question and
    answer that differ only in cells no query reads are duplicates;
  * the question is written compactly, at most 240 tokens (LLaVA/vicuna with
    BOS, and PaliGemma) for every parameter combination: one step per line as
    "a–b×c–d rows|columns", queried cells as "(r,c),(r,c)", the legend as
    "R red, ...";
  * easy asks THREE queried cells (n=4 grid, steps=1 reflection,
    minimum_hits=1), which gives easy 42,984 distinct items (answer-relevant D4
    classes, an exact enumeration).
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib

import numpy as np

from visro import api
from visro.seeding import canonical, digest

TASK = "mirroring_patterns"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

FONT = (_ROOT / 'assets/fonts/DejaVuSans.ttf')
FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"

# Generator colour index -> release code.  Index order is shared/puzzles/drawing.py COLORS /
# CODES ('.RBGYP'); only the white code changes.
CODES = "WRBGYP"
NAMES = {"R": "red", "B": "blue", "G": "green", "Y": "yellow", "P": "purple", "W": "white"}

# Exact per-tier parameters (read from mirrors_low.CONFIG and symbolic.mirrors; _load() asserts the
# generator still has them).
PARAMS = {
    "easy":   dict(n=4,  steps=1, nq=3,  minimum_hits=1, generator="generator/mirrors_low.py:build", first_axis="random", query_order="row-major"),
    "medium": dict(n=6,  steps=2, nq=3,  minimum_hits=1, generator="generator/mirrors_low.py:build", first_axis="random", query_order="row-major"),
    "hard":   dict(n=8,  steps=2, nq=4,  minimum_hits=2, generator="shared/puzzles/symbolic.py:mirrors", first_axis="rows", query_order="random"),
    "xhard":  dict(n=10, steps=4, nq=8,  minimum_hits=3, generator="shared/puzzles/symbolic.py:mirrors", first_axis="rows", query_order="random"),
    "xxhard": dict(n=12, steps=6, nq=12, minimum_hits=4, generator="shared/puzzles/symbolic.py:mirrors", first_axis="rows", query_order="random"),
}
_CFG_KEYS = ("n", "steps", "nq", "minimum_hits")
_COMMON = ("448x448 image; each cell i.i.d. one of R,B,G,Y,P (p=.88, uniform) else W; every rectangle spans "
           "rows r0+1..r1 and columns c0+1..c1 with r0,c0 < n/2 < r1,c1 (0-based half-open); "
           "operations alternate row/column reversal; whole-grid D4 compositions rejected; every queried cell "
           "reads a different initial cell, and its content passed through >= minimum_hits of the reflected "
           "rectangles (operations whose rectangle contained it when applied); answer != reading the queried "
           "cells off the initial grid")
TIER_DEFINITIONS = {
    "easy": "n=4 grid (88 px cells), steps=1 reflection (axis random), nq=3 queried cells in row-major order, "
            "minimum_hits=1 [generator/mirrors_low.py CONFIG] "
            "-- " + _COMMON,
    "medium": "n=6 grid (58 px cells), steps=2 reflections (first axis random), nq=3 queried cells in row-major "
              "order, minimum_hits=1 [generator/mirrors_low.py CONFIG] -- " + _COMMON,
    "hard": "n=8 grid (44 px cells), steps=2 reflections (first one reverses rows), nq=4 queried cells in random "
            "order, minimum_hits=2 [shared/puzzles/symbolic.py mirrors() rank 0] -- " + _COMMON,
    "xhard": "n=10 grid (35 px cells), steps=4 reflections (first one reverses rows), nq=8 queried cells in random "
             "order, minimum_hits=3 [shared/puzzles/symbolic.py mirrors() rank 1] -- " + _COMMON,
    "xxhard": "n=12 grid (29 px cells), steps=6 reflections (first one reverses rows), nq=12 queried cells in random "
              "order, minimum_hits=4 [shared/puzzles/symbolic.py mirrors() rank 2] -- " + _COMMON,
}

LEGEND = "R red, B blue, G green, Y yellow, P purple, W white"


def question(n: int, ops: list[dict], queries: list[list[int]]) -> str:
    """The release prompt.  ops/queries are 1-based and inclusive, as printed: one step per line,
    "a–b×c–d rows" or "a–b×c–d columns" (the rectangle rows a–b × columns c–d, and which order is
    reversed inside it), then the queried cells in answer order and the colour legend."""
    lines = [f"{o['rows'][0]}–{o['rows'][1]}×{o['cols'][0]}–{o['cols'][1]} "
             + ("rows" if o["axis"] == "rows" else "columns") for o in ops]
    cells = ",".join(f"({r},{c})" for r, c in queries)
    lead = ("Apply this step" if len(ops) == 1 else "Apply each step in order to the previous result")
    return (f"Cells are (row,column), numbered 1–{n} as labelled. {lead}: reverse the order of the given rows or "
            "columns inside the rows×columns rectangle.\n" + "\n".join(lines) + "\n"
            f"Then give the colors of {cells} in order as {len(queries)} letters, no separators: {LEGEND}.")


# --------------------------------------------------------------------------- generator loading

class _Gen:
    mirrors_low = None
    symbolic = None
    oracles = None
    builders = None


def _load() -> _Gen:
    if _Gen.mirrors_low is not None:
        return _Gen
    if hashlib.sha256(FONT.read_bytes()).hexdigest() != FONT_SHA256:
        raise RuntimeError(f"{FONT} is not the pinned DejaVuSans.ttf (sha256 {FONT_SHA256[:12]}...)")
    from shared.puzzles import builders, drawing, oracles, symbolic
    from tasks.planar_transformations.mirroring_patterns.generator import mirrors_low
    if drawing.CODES != ".RBGYP" or len(drawing.COLORS) != 6:
        raise RuntimeError("drawing colour table changed; re-check CODES mapping")
    for tier in ("easy", "medium"):
        cfg = mirrors_low.CONFIG[tier]
        if cfg != {k: PARAMS[tier][k] for k in _CFG_KEYS}:
            raise RuntimeError(f"mirrors_low.CONFIG[{tier}] changed: {cfg}")
    _Gen.mirrors_low, _Gen.symbolic, _Gen.oracles, _Gen.builders = mirrors_low, symbolic, oracles, builders
    return _Gen


# --------------------------------------------------------------------------- canonical key

def _final_letters(grid: list[str], ops0: list[list[int]]) -> np.ndarray:
    """Forward application of 0-based half-open ops [axis, r0, r1, c0, c1] to a letter grid."""
    g = np.array([list(row) for row in grid])
    for a, r0, r1, c0, c1 in ops0:
        g[r0:r1, c0:c1] = np.flip(g[r0:r1, c0:c1], axis=a)
    return g


def _d4_variants(n: int, ops: list[list[int]], cells: list[tuple[int, int, str]]):
    """All 8 whole-grid symmetric copies of (ops, queried cells with their answer letters); ops are
    0-based [axis, r0, r1, c0, c1] half-open, cells 0-based (row, col, letter).  A symmetric copy is
    the same puzzle drawn rotated/mirrored, with the same answer letters at the mapped cells."""
    for transpose in (False, True):
        for flip_r in (False, True):
            for flip_c in (False, True):
                o = [[1 - a, c0, c1, r0, r1] if transpose else [a, r0, r1, c0, c1] for a, r0, r1, c0, c1 in ops]
                q = [(c, r, x) if transpose else (r, c, x) for r, c, x in cells]
                if flip_r:
                    o = [[a, n - r1, n - r0, c0, c1] for a, r0, r1, c0, c1 in o]
                    q = [(n - 1 - r, c, x) for r, c, x in q]
                if flip_c:
                    o = [[a, r0, r1, n - c1, n - c0] for a, r0, r1, c0, c1 in o]
                    q = [(r, n - 1 - c, x) for r, c, x in q]
                yield o, sorted(q)


def core_key(n: int, grid: list[str], ops0: list[list[int]], queries0: list[tuple[int, int]]) -> str:
    """sha256 over the lexicographically smallest D4 copy of (operations, sorted queried cells each
    with the letter it ends up with).

    Only ANSWER-RELEVANT content is keyed: the colours of cells that no query reads are left out, so
    two items with the same question and answer over grids that differ elsewhere are duplicates.
    Query ORDER is excluded too: the same cells listed in another order is the same puzzle with a
    permuted answer string.
    """
    fin = _final_letters(grid, ops0)
    cells = [(int(r), int(c), str(fin[r, c])) for r, c in queries0]
    best = min(canonical([o, [list(x) for x in q]]) for o, q in _d4_variants(n, ops0, cells))
    return digest({"task": TASK, "n": n, "answer_relevant_core": best})


# --------------------------------------------------------------------------- build

def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in PARAMS:
        raise ValueError(f"unknown tier {tier!r}")
    p = PARAMS[tier]
    G = _load()
    if tier in G.mirrors_low.CONFIG:
        rec = G.mirrors_low.build(tier, seed, index)
        image = G.mirrors_low.panel(rec["meta"]["grid"])
    else:
        rec, image = G.builders.build(TASK, tier, seed, index, G.symbolic.mirrors)
    # The builders' own gate (shared/puzzles/oracles.py check): backward trace of every query,
    # minimum trajectory depth, distinct sources.  Raises AssertionError.
    G.oracles.check(rec)
    m = rec["meta"]
    n = len(m["grid"])
    if m.get("family") != "partial_reflections" or n != p["n"] or len(m["operations"]) != p["steps"] \
            or len(m["queries"]) != p["nq"] or m["minimum_query_operations"] != p["minimum_hits"]:
        raise RuntimeError(f"generator output does not match the {tier} parameters")
    grid = ["".join(CODES[v] for v in row) for row in m["grid"]]
    final = ["".join(CODES[v] for v in row) for row in m["final_grid"]]
    ops0 = [list(map(int, op)) for op in m["operations"]]
    q0 = [(int(r), int(c)) for r, c in m["queries"]]
    answer = "".join(final[r][c] for r, c in q0)
    if answer != rec["answer"].replace(".", "W"):
        raise RuntimeError("generator answer disagrees with its own final grid")
    # Gate for every tier (mirrors_low.py also applies it at easy/medium).
    if answer == "".join(grid[r][c] for r, c in q0):
        raise api.Reject("answer equals the initial-grid reading of the queried cells")
    ops = [{"axis": "rows" if a == 0 else "cols", "rows": [r0 + 1, r1], "cols": [c0 + 1, c1]}
           for a, r0, r1, c0, c1 in ops0]
    queries = [[r + 1, c + 1] for r, c in q0]
    text = question(n, ops, queries)
    image = image.convert("RGB")
    image.info.clear()
    cell = 352 // n
    meta = {
        "family": "partial_reflections",
        "n": n,
        "grid": grid,                 # rows, top to bottom; letters as in the question legend
        "operations": ops,            # 1-based inclusive, as printed in the question
        "queries": queries,           # 1-based (row, column), in the asked order
        "final_grid": final,
        "minimum_query_operations": p["minimum_hits"],
        "requirements": {k: p[k] for k in _CFG_KEYS},
        "codes": NAMES,
        "render": {"size": list(image.size), "x0": (image.width - n * cell) // 2, "y0": 48, "cell": cell,
                   "label_font": "DejaVuSans 12px", "outline_rgb": [120, 125, 132]},
        "generator": p["generator"],
        "first_axis": p["first_axis"],
        "query_order": p["query_order"],
        "attempts": m.get("attempts"),
    }
    return api.Item(question=text, answer=answer, image=image, meta=meta,
                    core_key=core_key(n, grid, ops0, q0), answer_type="text", choices=None)
