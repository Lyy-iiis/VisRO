"""paper_folding: staged fold-and-punch puzzles answered as a queried-cell bitstring.

Generator routes:

  easy / medium / hard  generator/paper_query.py  (enumerated core pool, affine-permuted)
  xhard / xxhard        folding.py (random fold plan + punches)

Both routes share folding.py's physics (fold_plan / advance / safe_centers) and
its renderer (draw_packet on shared/puzzles/drawing.py's canvas).  This adapter wraps that
code unchanged, and:

* Fonts: shared/puzzles/drawing.py draws with the repo's
  assets/fonts/DejaVuSans.ttf, which this adapter sha-pins at import, so there
  is no host-font fallback.
* Seeds: the framework's derived seed (task, tier, split, base seed, index) is
  passed as the generator seed, so every random stream (pool permutation, query
  sampling, fold/punch sampling) is keyed by tier and split.  For the low tiers
  this turns the per-seed affine permutation into a uniform draw from the core
  pool; duplicate cores are then removed by the driver's core_key dedup.
* Gates: the per-record gates are applied here (paper_query contract and
  question regeneration, oracles.check_paper reverse-unfold oracle, check_render
  size and label profile, verify_punch_pixels).  The release gate proper is the
  independent verifier (verifier.py), which also validates the question text.
* Queried cells.  Holes come in groups that share a packet position after a
  punch-stage fold ("mates", computable from the fold text), so the query list
  is drawn by _matched_queries: one exclusive hole per punch stage, exactly nq/2
  ones, and a draw is accepted only when the multiset of per-cell text
  signatures (mate counts at every punch stage + on-diagonal-crease flag) is
  identical over the 1-cells and the 0-cells.  The core (folds, punches), and
  hence the image and core_key, is the generator's.
* Question.  Rebuilt from the record's structure (n, folds, punch stages,
  queried cells) by question_text: one compact template for all five tiers, at
  most 240 tokens on every tier, worst case included (LLaVA vicuna-7b-v1.5 +
  BOS and PaliGemma), with every convention stated once.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import copy
import hashlib
from pathlib import Path

import numpy as np

from visro import api
from visro.seeding import digest, plain, seed_text

TASK = "paper_folding"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

TIER_DEFINITIONS = {
    "easy": "paper_query: 8x8 sheet; 1 axis fold (v/h at the packet midline, either direction); "
            "punch stage after fold 1 with 4 punches (panel P1); 6 queried cells -> 6-digit bitstring "
            "(exactly 3 ones, structure-matched query sampler); core drawn uniformly from the enumerated "
            "143,840-core pool",
    "medium": "paper_query: 8x8 sheet; 2 axis folds; punch stages after folds 1 and 2 with 2+2 "
              "punches (panels P1,P2); 8 queried cells -> 8-digit bitstring (exactly 4 ones, structure-matched "
              "query sampler); core drawn uniformly from the enumerated 470,400-core pool",
    "hard": "paper_query: 8x8 sheet; 2 axis folds + 1 diagonal fold of the 4x4 packet (d or a); "
            "punch stages after folds 1 and 3 with 2+2 punches (panels P1,P2; P2 triangular); 10 queried "
            "cells -> 10-digit bitstring (exactly 5 ones, structure-matched query sampler); core drawn "
            "uniformly from the enumerated 112,320-core pool",
    "xhard": "folding: 16x16 sheet; 4 axis folds, no diagonal; punch stages after folds "
             "2, 3 and 4 with 1+1+2 punches (panels P1,P2,P3); 12 queried cells -> 12-digit bitstring "
             "(exactly 6 ones, structure-matched query sampler); random fold plan and punches, <=500 attempts",
    "xxhard": "folding: 16x16 sheet; 4 axis folds + 1 diagonal fold of the 4x4 packet "
              "(d or a); punch stages after folds 2, 4 and 5 with 1+2+2 punches (panels P1,P2,P3; P3 "
              "triangular); 16 queried cells -> 16-digit bitstring (exactly 8 ones, structure-matched query "
              "sampler); random fold plan and punches, <=500 attempts",
}

# Exact tier parameters (the verifier keeps its own copy and checks records against it).
TIER_PARAMS = {
    "easy": dict(n=8, folds=1, stages=[1], punches=[4], queries=6, diagonal_last=False),
    "medium": dict(n=8, folds=2, stages=[1, 2], punches=[2, 2], queries=8, diagonal_last=False),
    "hard": dict(n=8, folds=3, stages=[1, 3], punches=[2, 2], queries=10, diagonal_last=True),
    "xhard": dict(n=16, folds=4, stages=[2, 3, 4], punches=[1, 1, 2], queries=12, diagonal_last=False),
    "xxhard": dict(n=16, folds=5, stages=[2, 4, 5], punches=[1, 2, 2], queries=16, diagonal_last=True),
}
LOW_TIERS = ("easy", "medium", "hard")

FONT = (_ROOT / 'assets/fonts/DejaVuSans.ttf')
FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"

CANVAS = 448


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_pins() -> None:
    if _sha(FONT) != FONT_SHA256:
        raise RuntimeError(f"{FONT} is not the pinned DejaVuSans.ttf")


_check_pins()

from shared.puzzles import drawing, oracles  # noqa: E402
from tasks.spatial_transformations.paper_folding.generator import folding, paper_query  # noqa: E402


def _gate_low(tier, rec):
    """The per-record gate of the paper_query tiers."""
    m = rec["meta"]
    cfg = paper_query.CONFIG[tier]
    assert m["requirements"] == cfg and m["n"] == cfg["n"] and len(m["folds"]) == cfg["folds"], "contract"
    assert [e["stage"] for e in m["events"]] == cfg["stages"], "stages"
    assert [len(e["centers"]) for e in m["events"]] == cfg["punches"], "punch counts"
    assert rec["question"] == paper_query.question(m), "question differs from paper_query.question(meta)"
    oracles.check(rec)


def _gate_render(rec, im):
    """The render gate of every paper_folding tier."""
    assert im.mode == "RGB" and im.size == (CANVAS, CANVAS), (im.mode, im.size)
    oracles.check_render(rec, im)


def _generate(tier: str, seed: int, index: int):
    if tier in LOW_TIERS:
        try:
            rec = paper_query.build(tier, seed, index)
        finally:
            paper_query.permutation.cache_clear()  # one entry per derived seed otherwise
        _gate_low(tier, rec)
        im = paper_query.render(rec)          # includes verify_punch_pixels
        generator, template = "generator/paper_query.py:build", "paper_query"
    else:
        rng = drawing.rng_for(TASK, tier, seed, index)  # exactly shared/puzzles/builders.py build()'s stream
        rec, im = folding.paper_folding(tier, rng)
        rec["builder_family"] = rec["meta"]["family"]
        rec["image_labels"] = im.info.get("diagram_labels", [])
        oracles.check(rec)
        generator, template = "generator/folding.py:paper_folding", "folding"
    _gate_render(rec, im)
    return rec, im, generator, template


# --------------------------------------------------------------------------- query sampler

# Per item; a core with no matched draw within this many draws is rejected (api.Reject).
MAX_QUERY_DRAWS = 1 << 20
_BATCH = 2048


def _stage_structure(n: int, folds, events, truth):
    """Per punch stage: packet position id of every original cell (index r*n + c);
    each stage's exclusive holes; the on-diagonal-crease flag; the hole mask."""
    pts = np.array([(2 * c + 1, 2 * r + 1) for r in range(n) for c in range(n)])
    maps = []
    for step in folds:
        pts = folding.advance(pts, step)
        maps.append(pts.copy())
    stages = [e["stage"] for e in events]
    pos = np.stack([maps[s - 1][:, 0] * (4 * n + 1) + maps[s - 1][:, 1] for s in stages])
    effects = []
    for s, e in zip(stages, events):
        hit = np.zeros(n * n, bool)
        for x, y in e["centers"]:
            hit |= (maps[s - 1][:, 0] == x) & (maps[s - 1][:, 1] == y)
        effects.append(hit)
    holes = np.asarray(truth, bool).reshape(-1)
    assert np.array_equal(np.any(effects, axis=0), holes), "truth != union of stage effects"
    exclusive = [np.flatnonzero(eff & ~np.any([o for j, o in enumerate(effects) if j != i] or [np.zeros_like(eff)],
                                               axis=0)) for i, eff in enumerate(effects)]
    assert all(len(x) for x in exclusive), "a punch stage owns no exclusive hole"
    last = folds[-1]
    diag = np.zeros(n * n, bool)
    if last["axis"] in ("d", "a") and stages[-1] == len(folds):
        x, y = maps[-1][:, 0], maps[-1][:, 1]
        diag = (x - y == last["line"]) if last["axis"] == "d" else (x + y == last["line"])
    return pos, exclusive, diag, holes


def signatures(cells, pos, diag):
    """Text-computable signature of each queried cell: for every punch stage the number
    of OTHER queried cells at the same packet position, plus the on-crease flag.
    cells: (..., nq) int array -> (..., nq) int codes."""
    nq = cells.shape[-1]
    p = pos[:, cells]                                            # (S, ..., nq)
    cnt = (p[..., :, None] == p[..., None, :]).sum(-1) - 1        # (S, ..., nq)
    code = diag[cells].astype(np.int64)
    for s in range(pos.shape[0]):
        code = code * nq + cnt[s]
    return code


def _subsets(rng, ids, size, batch):
    """`batch` uniform random `size`-subsets of `ids` (ascending), via random keys."""
    if size == 0:
        return np.empty((batch, 0), ids.dtype)
    keys = rng.random((batch, len(ids)))
    return np.sort(ids[np.argpartition(keys, size - 1, axis=1)[:, :size]], axis=1)


def _balanced_rows(cells, pos, diag, k):
    """Rows of `cells` (B, 2k; ones first) whose 1-cells and 0-cells have the same
    signature multiset.  Filters stage by stage (a projection of the signature must
    already balance), which accepts exactly the rows the full comparison accepts."""
    nq = cells.shape[1]
    rows = np.arange(len(cells))
    code = diag[cells].astype(np.int64)
    for s in [None, *range(pos.shape[0] - 1, -1, -1)]:   # crease flag, then the last stage first
        if s is not None:
            p = pos[s][cells]
            code = code * nq + (p[:, :, None] == p[:, None, :]).sum(-1) - 1
        ok = (np.sort(code[:, :k], 1) == np.sort(code[:, k:], 1)).all(1)
        rows, cells, code = rows[ok], cells[ok], code[ok]
        if not len(rows):
            break
    return rows


def _matched_queries(n, folds, events, truth, nq, rng):
    """Draw (ones, zeros) as the generator does -- one exclusive hole per punch stage,
    the other ones uniform over the remaining holes, zeros uniform over intact cells --
    and accept the first draw whose 1-cells and 0-cells carry the same multiset of
    signatures.  Returns (shuffled 0-based (row, col) list, draws used)."""
    pos, exclusive, diag, holes = _stage_structure(n, folds, events, truth)
    k = nq // 2
    hole_ids, intact_ids = np.flatnonzero(holes), np.flatnonzero(~holes)
    S = len(exclusive)
    assert S <= k <= len(hole_ids) and k <= len(intact_ids), "not enough holes / intact cells"
    for start in range(0, MAX_QUERY_DRAWS, _BATCH):
        first = np.stack([ex[rng.integers(len(ex), size=_BATCH)] for ex in exclusive], 1)       # (B, S)
        keys = rng.random((_BATCH, len(hole_ids)))
        keys[(hole_ids[None, :, None] == first[:, None, :]).any(-1)] = 2.0                       # already taken
        extra = (np.sort(hole_ids[np.argpartition(keys, k - S - 1, axis=1)[:, :k - S]], axis=1) if k > S
                 else np.empty((_BATCH, 0), hole_ids.dtype))
        zeros = _subsets(rng, intact_ids, k, _BATCH)
        cells = np.concatenate([first, extra, zeros], 1)                                        # (B, nq)
        rows = _balanced_rows(cells, pos, diag, k)
        if len(rows):
            j = int(rows[0])
            chosen = cells[j][rng.permutation(nq)]
            return [(int(c) // n, int(c) % n) for c in chosen], start + j + 1
    raise api.Reject(f"no structure-matched query list in {MAX_QUERY_DRAWS} draws")


# --------------------------------------------------------------------------- question text

# one compact template for all tiers: at most 240 tokens under the LLaVA (vicuna + BOS) and
# PaliGemma tokenizers, worst case over every plan and query list.  It states every convention once:
#   n x n grid; (row,column) cells numbered from 1 downward/rightward in the original sheet
#     ("Original nxn grid sheet: cells (row,column) from 1 down/right;");
#   crease x/y values are original grid lines, 0 at the top-left corner, and a fold names
#     its crease as x<k> / y<k> ("crease x/y = grid line (0 at top left)."; "x8");
#   fold the WHOLE packet, in order, without rotating;
#   each fold names the half that MOVES ("moving the named half: x8 left" = the half left
#     of the crease x=8 is folded over onto the right half, keep="high").  The destination
#     needs no words: fold_plan puts every axis crease
#     on the current packet's midline and every diagonal crease on the square packet's
#     diagonal, so the moving half always lands exactly on the other half;
#   the punch panel right after its fold (", punch Pk"); fully unfold at the end;
#   panels in the original-sheet frame and grid (unchanged grid squares), each zoomed on its
#     own; red circles are small holes through ALL current layers; earlier holes are not
#     redrawn ("omitted");
#   the answer is one K-digit string without spaces, one digit per listed cell in order,
#     1 = a hole centred in it, else 0.
# The verifier keeps its own copy and requires byte-exact re-emission.

_MOVING_HALF = {"v": ("right", "left"), "h": ("bottom", "top")}      # [keep == "low", keep == "high"]
_MOVING_TRIANGLE = {"d": "upper right triangle", "a": "upper left triangle"}


def _fold_text(axis: str, line2, keep: str) -> str:
    """One fold: crease (doubled units -> grid-line value) and the half that moves."""
    if axis in _MOVING_HALF:
        return f"{'x' if axis == 'v' else 'y'}{line2 / 2:g} {_MOVING_HALF[axis][keep != 'low']}"
    return _MOVING_TRIANGLE[axis]


def question_text(n: int, folds, stages, queries) -> str:
    """folds: [(axis, line2, keep)]; stages: the fold each panel P1..Pk follows, in panel
    order; queries: 0-based (row, col) in answer order."""
    steps = "; ".join(_fold_text(*f) + (f", punch P{stages.index(i) + 1}" if i in stages else "")
                      for i, f in enumerate(folds, 1))
    panels = "Panel P1 shows" if len(stages) == 1 else "Each panel shows"
    cells = ",".join(f"({r + 1},{c + 1})" for r, c in queries)
    return (f"Fold the {n}x{n} grid sheet in order without turning it, moving the named half (crease x/y = grid line, "
            f"0 at top left): {steps}; then unfold. {panels} the packet in the original frame with red holes punched "
            f"through all layers (earlier holes omitted). Give one {len(queries)}-digit string: per cell (row,column) "
            f"below in order, 1 if a hole is centred there, else 0: {cells}.")


def release_question(meta: dict, queries) -> str:
    """The release question for a generator record's meta and a query list."""
    return question_text(meta["n"], [(f["axis"], f["line"], f["keep"]) for f in meta["folds"]],
                         [e["stage"] for e in meta["events"]], queries)


def _query_rng(tier: str, seed: int, index: int):
    key = f"{seed_text(491310769518)}/{TASK}/{seed_text(150812319849027196635146609136663230001)}/{tier}/{int(seed)}/{int(index)}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def core_key_of(meta: dict) -> str:
    """The physical puzzle: sheet size, fold plan and staged punch sets.

    The image is a function of exactly these, so two records with the same key
    show the same picture; the queried cells are deliberately left out (a second
    query list over the same picture is a near-duplicate, not a new item)."""
    core = dict(n=meta["n"],
                folds=[[f["axis"], f["line"], f["keep"]] for f in meta["folds"]],
                events=[[e["stage"], sorted(map(list, e["centers"]))] for e in meta["events"]])
    return digest(dict(task=TASK, core=core))


def _requery(tier: str, seed: int, index: int, rec: dict):
    """Replace the generator's query list by a structure-matched one (see module doc).
    Returns the release record and the number of query draws."""
    m = rec["meta"]
    nq = TIER_PARAMS[tier]["queries"]
    queries, draws = _matched_queries(m["n"], m["folds"], m["events"], m["truth"], nq, _query_rng(tier, seed, index))
    new = dict(rec)
    new["meta"] = dict(m, queries=[[r, c] for r, c in queries])
    new["answer"] = "".join(str(m["truth"][r][c]) for r, c in queries)
    new["question"] = release_question(m, queries)
    oracles.check(new)  # reverse-unfold oracle: answer, distinct in-sheet cells, every stage covered
    return new, draws


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in TIER_PARAMS:
        raise ValueError(f"unknown tier {tier!r}")
    rec, im, generator, template = _generate(tier, int(seed), int(index))
    rec, draws = _requery(tier, int(seed), int(index), rec)
    m = rec["meta"]
    p = TIER_PARAMS[tier]
    meta = dict(
        family=m["family"],
        generator=generator,
        template=template,
        question_form="staged_paper_punches/bitstring",
        coordinates="doubled original-sheet units: cell (row r, col c), 0-based, has centre (2c+1, 2r+1); "
                    "grid line x=k is 2k; queries are 0-based [row, col]",
        tier_params=dict(p),
        n=m["n"],
        folds=copy.deepcopy(m["folds"]),
        events=copy.deepcopy(m["events"]),
        truth=m["truth"],
        queries=m["queries"],
        query_sampler=dict(name="structure-matched/v1", draws=draws),
        render=m.get("render"),
        image_labels=list(rec.get("image_labels") or []),
        canvas=[CANVAS, CANVAS],
    )
    if tier in LOW_TIERS:
        meta.update(core_rank=m["core_rank"], core_capacity=m["core_capacity"])
    else:
        meta.update(attempts=m["attempts"])
    meta = plain(meta)
    return api.Item(question=rec["question"], answer=rec["answer"], image=im, meta=meta,
                    core_key=core_key_of(meta), answer_type="bitstring", choices=None)
