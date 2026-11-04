"""jigsaw_completion: fill connected holes in an edge-strip board from a candidate pool.

Generator:

    easy, medium   generator/jigsaw_joint_low.py  build(tier, seed, index) + render(rec)
    hard..xxhard   shared/puzzles/builders.py build() -> generator/board_geometry.py jigsaw()

Wiring (the generator's logic is not changed):
  * seed: the framework's derived seed (already keyed by task, tier, split,
    base seed and index) is passed as the generator's `seed`.
  * fonts: the renderers use the committed assets/fonts/DejaVuSans.ttf
    (sha-checked), never a system fallback, so the PNGs are reproducible.
  * the generator's own gates run on every candidate: jigsaw_joint_low.check
    (inside build) and check_render for easy/medium, and shared/puzzles/oracles.py check()
    for every tier.
  * question: after the generator's own gates have passed, the question is
    replaced by QUESTION below, which states the conventions the answer depends
    on (the letter above a candidate names it, outer-border sides are
    unconstrained, one pair per hole) and whose format illustration X1,Y0,Z3
    uses letters no pool contains (pools stop at J); any gold that shares a pair
    or a letter with it is rejected.

core_key is the D4-canonical *reasoning* key: per hole and per side, what the
hole's piece must match there (a visible edge, the outer boundary, or another
hole), plus the candidate pool as a multiset of rotation classes.  Decorative
board tiles that touch no hole, the candidates' display order and display
rotations, and whole-puzzle rotations/reflections do not change it.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

# Generation runs one process per worker; OpenBLAS (pulled in by numpy) would otherwise
# start one busy-waiting thread per core in every worker.  Set before numpy is first
# imported; an explicit environment setting wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from visro import api  # noqa: E402
from visro.seeding import digest  # noqa: E402

TASK = "jigsaw_completion"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

FONT = (_ROOT / 'assets/fonts/DejaVuSans.ttf')
FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"

TIERS_PARAMS = {  # n, holes, ports, pool, canvas -- restated for meta/core_key only
    "easy": (3, 2, 2, 4, 448),
    "medium": (3, 2, 2, 5, 448),
    "hard": (3, 2, 2, 6, 448),
    "xhard": (4, 3, 3, 8, 588),
    "xxhard": (5, 5, 4, 10, 588),
}

TIER_DEFINITIONS = {
    "easy": ("n=3 board, holes=2 (one connected region), ports=2 strips per edge, 5 colours, pool=4 (2 true + 2 "
             "decoys: one port of a side facing another cell recoloured, parents dealt round-robin over the holes), "
             "random display quarter-turns, exactly 1 joint solution; canvas 448 px, board tile 86 px, candidate "
             "72 px in one row [generator/jigsaw_joint_low.py CONFIG easy]"),
    "medium": ("n=3 board, holes=2 (connected), ports=2, 5 colours, pool=5 (2 true + 3 recolour decoys, round-robin "
               "parents), random display quarter-turns, exactly 1 joint solution; canvas 448 px, board tile 86 px, "
               "candidate 64 px in one row [generator/jigsaw_joint_low.py CONFIG medium]"),
    "hard": ("n=3 board, holes=2 (connected), ports=2, 5 colours (same board, holes and ports as easy/medium), "
             "pool=6 (2 true + 4 decoys: each a true tile drawn independently with replacement, with one non-uniform "
             "side's ports cyclically shifted by one, any side; a true tile that parents no decoy has no one-edge "
             "near-twin in the pool), random display quarter-turns, exactly 1 joint solution; canvas 448 px, board "
             "tile 78 px, candidate 54 px in two rows (5 + 1) [board_geometry.jigsaw() rank 0]"),
    "xhard": ("n=4 board, holes=3 (connected), ports=3, 5 colours, pool=8 (3 true + 5 port-shift decoys drawn as in "
              "hard, independent parents), random display quarter-turns, exactly 1 joint solution; canvas 588 px, "
              "board tile 80 px, candidate 70 px in two rows (5 + 3) [board_geometry.jigsaw() rank 1]"),
    "xxhard": ("n=5 board, holes=5 (connected), ports=4, 5 colours, pool=10 (5 true + 5 port-shift decoys drawn as "
               "in hard, independent parents), random display quarter-turns, exactly 1 joint solution; canvas 588 px, "
               "board tile 64 px, candidate 70 px in two rows (5 + 5) [board_geometry.jigsaw() rank 2]"),
}

# The question every tier asks.  It replaces the generator's fixed wording
# (see the module docstring).  The format illustration uses letters no pool
# contains, so it can never equal or share a pair with a gold.
FORMAT_EXAMPLE = "X1,Y0,Z3"
QUESTION = (
    'Fill each numbered gray hole with a lettered piece so colored strips continue across every '
    'shared edge, including between holes; board-border sides are free. Pieces turn clockwise in '
    'quarter turns, never flip, and are used at most once. Answer letter+turns (0-3, from as '
    'drawn) per hole in hole order, comma-separated, e.g. '
    f"{FORMAT_EXAMPLE}."
)

LETTERS = "ABCDEFGHIJ"
DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))  # sides 0..3 = top, right, bottom, left (clockwise)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Gen:
    """The generator modules, imported once per process."""

    def __init__(self):
        if _sha(FONT) != FONT_SHA256:
            raise RuntimeError(f"jigsaw_completion: {FONT} is not the pinned DejaVuSans")
        from shared.puzzles import builders, oracles
        from tasks.planar_transformations.jigsaw_completion.generator import board_geometry, jigsaw_joint_low
        self.builders = builders
        self.oracles = oracles
        self.geometry = board_geometry
        self.low = jigsaw_joint_low
        if set(self.low.CONFIG) != {"easy", "medium"}:
            raise RuntimeError("jigsaw_joint_low.CONFIG no longer covers exactly easy/medium")


@lru_cache(None)
def _gen() -> _Gen:
    return _Gen()


_gen()  # import before the driver forks its workers


# ----------------------------------------------------------------- core_key
def _turn(tile, k):
    """Clockwise quarter-turns of a clockwise edge list (generator convention)."""
    return tuple(tuple(tile[(i - k) % 4]) for i in range(4))


def _mirror(tile):
    """Left-right mirror image of a clockwise edge list."""
    t, r, b, l = (tuple(e) for e in tile)
    return (t[::-1], l[::-1], b[::-1], r[::-1])


def _rot_class(tile):
    return min(_turn(tile, k) for k in range(4))


def _reasoning_key(n, board, holes, pool):
    holes = sorted(holes)
    hs = set(holes)
    sides = []
    for (r, c) in holes:
        row = []
        for s, (dr, dc) in enumerate(DELTA):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n and 0 <= cc < n):
                row.append("B")
            elif (rr, cc) in hs:
                row.append(["H", holes.index((rr, cc))])
            else:
                row.append(list(board[rr][cc][(s + 2) % 4]))
        sides.append(row)
    return [sides, sorted(_rot_class(p) for p in pool)]


def _transform(n, board, holes, pool, k, flip):
    """Rotate the whole puzzle k quarter-turns clockwise, after an optional mirror."""
    b = [[tuple(map(tuple, t)) if t is not None else None for t in row] for row in board]
    h = [tuple(x) for x in holes]
    p = [tuple(map(tuple, t)) for t in pool]
    if flip:
        b = [[_mirror(b[r][n - 1 - c]) if b[r][n - 1 - c] is not None else None for c in range(n)] for r in range(n)]
        h = [(r, n - 1 - c) for r, c in h]
        p = [_mirror(t) for t in p]
    for _ in range(k):
        # (r, c) -> (c, n-1-r); new cell (R, C) came from (n-1-C, R)
        b = [[_turn(b[n - 1 - C][R], 1) if b[n - 1 - C][R] is not None else None for C in range(n)] for R in range(n)]
        h = [(c, n - 1 - r) for r, c in h]
    return b, h, p


def core_key_of(meta) -> str:
    n = len(meta["board"])
    holes = [tuple(x) for x in meta["holes"]]
    board = [[None if (r, c) in holes else meta["board"][r][c] for c in range(n)] for r in range(n)]
    keys = []
    for flip in (False, True):
        for k in range(4):
            b, h, p = _transform(n, board, holes, meta["pool"], k, flip)
            keys.append(json.dumps(_reasoning_key(n, b, h, p), separators=(",", ":")))
    return digest({"task": TASK, "reasoning": min(keys)})


def example_overlap(answer: str) -> list[str]:
    """What the gold shares with FORMAT_EXAMPLE: equality, a pair, a letter, a prefix."""
    ex, gold = FORMAT_EXAMPLE.split(","), answer.split(",")
    out = []
    if answer == FORMAT_EXAMPLE or FORMAT_EXAMPLE.startswith(gold[0]) or answer.startswith(ex[0]):
        out.append("prefix")
    out += [f"pair {p}" for p in sorted(set(ex) & set(gold))]
    out += [f"letter {L}" for L in sorted({p[0] for p in ex} & {p[0] for p in gold})]
    return out


# ----------------------------------------------------------------- build
def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in TIERS_PARAMS:
        raise ValueError(f"unknown tier {tier!r}")
    g = _gen()
    if tier in g.low.CONFIG:
        rec = g.low.build(tier, seed, index)          # runs jigsaw_joint_low.check(rec) (incl. oracles.check)
        im = g.low.render(rec)
        g.low.check_render(rec, im)                   # strip pixels == meta
        generator = "generator/jigsaw_joint_low.py:build"
    else:
        rec, im = g.builders.build(TASK, tier, seed, index, g.geometry.jigsaw)
        g.oracles.check(rec)                          # independent joint solver (shared/puzzles/oracles.py)
        generator = "generator/board_geometry.py:jigsaw"
    m = rec["meta"]
    n, nh, ports, npool, canvas = TIERS_PARAMS[tier]
    if (len(m["board"]), len(m["holes"]), m["ports"], len(m["pool"]), im.width) != (n, nh, ports, npool, canvas):
        raise api.Reject("generator output does not match the tier row")
    answer = rec["answer"]
    shared = example_overlap(answer)
    if shared:
        raise api.Reject(f"gold shares {shared} with the question's format example")
    if m.get("family") != "joint_edge_jigsaw":
        raise api.Reject("unexpected generator family")
    meta = {
        "family": m["family"],
        "generator": generator,
        "n": n,
        "ports": m["ports"],
        "holes": [list(h) for h in m["holes"]],
        "board": m["board"],          # holes hold the hidden true tile
        "pool": m["pool"],            # candidates A, B, ... in display order and display rotation
        "solution": [list(s) for s in m["solution"]],
        "edge_encoding": "each tile = 4 edges clockwise from the top (top, right, bottom, left); each edge lists "
                         "its strip colours in clockwise order; colours 1..5 = red, blue, green, yellow, purple",
        "canvas": im.width,
        "attempts": m.get("attempts"),
        "generator_record_id": rec.get("id"),
    }
    return api.Item(
        question=QUESTION,
        answer=answer,
        image=im,
        meta=meta,
        core_key=core_key_of(meta),
        answer_type="token-list",
    )
