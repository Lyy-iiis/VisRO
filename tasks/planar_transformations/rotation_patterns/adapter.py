"""rotation_patterns: the five-rung rotation-pattern ladder.

Generator: generator/ladder.py; this adapter calls ladder.build(seed, idx, tier)
unchanged: the TIERS table, the word pools, every gate (S1 S2 S4 G1 G3 G4 G5 G6
G8 G11 G12 G13 G15..G19, W1..W8), the wording (question.py) and the drawing
(dial.py) are the generator's own.  The question carries no worked format
example.

What the adapter adds
---------------------
* Seeding.  `seed` is visro's derived seed, already keyed by (task, tier, split,
  base seed, index); it becomes the `seed` of ladder.build, whose rng key is
  "<tier rngkey>:<seed>:<idx>", so no two tiers or splits share a stream.  idx is
  the candidate index, which also fixes easy's source (idx % 16 == 0 -> L=3).
* Figure-level dedup.  ladder.build draws a word and then a random member of its
  rotation / mirror / level-inversion class.  visro builds items statelessly in
  parallel, so the "never repeat a FIGURE" rule lives in core_key: core_key is
  the necklace (min cyclic rotation) of the base word plus L.  Two items whose
  base figures are the same dial at any rotation have the same core_key and the
  driver keeps only the first, within a cell and across the task's tiers.
* Pools.  The by-SPEC pools are read from generator/pools/ (*.json.gz,
  md5-asserted against word_pools.SPEC on load), so a fresh process never pays
  the 20-60 min cold rebuild.
* Font.  The index badges are rasterised with assets/fonts/DejaVuSans-Bold.ttf
  (dial.py loads it through shared/drawing/fonts.py); the adapter checks that
  file against its pinned md5 before the first build.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib

from visro import api
from visro.seeding import digest

from tasks.planar_transformations.rotation_patterns.generator import ladder, word_pools as wp

TASK = "rotation_patterns"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

FONT = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')
FONT_MD5 = "21546d802d508f3d358082d85bc0d9f1"  # the pinned badge font

_COMMON = ("16 spokes on the 22.5 deg lattice, bead radius = level (rails 34..67 px in a 168 px figure box); "
           "bank = distinct clockwise shifts from the 14 multiples of 22.5 deg other than 0 and 180; "
           "gates G1 worst changed blob >=110 px, G5 rotation fidelity <=0.045, G8 level legibility, "
           "pool gates W1..W8; centroid probe recorded in meta (probe_m1), not gated; answer tokens comma-separated "
           "in question order; the question prints no format example")

TIER_DEFINITIONS = {
    "easy": ("L=3 bead lengths (16.50 px step, _pool_L3w8full 1,553 words md5 ff47aadd) when index%16==0, "
             "else L=4 (11.00 px step, _pool_L4w8full 34,811 words md5 2b69aa46); strip6 canvas 1232x224 = "
             "base + 6 numbered candidates; base and all 6 candidates pairwise >=45 deg apart (min_gap=2); "
             "2 present absolute questions, plain body; answer = 2 distinct figure numbers in 1..6; "
             + _COMMON),
    "medium": ("L=5 (8.25 px step, _pool_L5m 126,936 words md5 94560be3: hard's L=5 W1..W7 family enumerated "
               "exhaustively, W8 ideal centroid <=0.17 px for this tier); strip6 canvas 1232x224 = base + "
               "6 numbered candidates on the full 22.5 deg lattice (min_gap=1); 3 present absolute questions, "
               "plain body; answer = 3 distinct figure numbers in 1..6; " + _COMMON),
    "hard": ("L=5 (8.25 px step, _pool_L5w8 35,771 words md5 ad8748ba); wide12 canvas 1232x364 = base + 12 "
             "numbered candidates (min_gap=1); 9 present absolute questions, plain body; answer = 9 distinct "
             "figure numbers in 1..12; " + _COMMON),
    "xhard": ("L=5 (8.25 px step, _pool_L5xwd near-twin family: d(8) in {6,8,10} and the 180 deg twin is the "
              "closest rotation, 17,565 words md5 6a813134); wide12 canvas 1232x364 = base + 12 candidates "
              "(min_gap=1); enumerated body with 12 questions, one angle per line = "
              "10 present absolute + the 2 missing angles (answer none); answer = 12 tokens in 1..12|none; "
              + _COMMON),
    "xxhard": ("L=5 (8.25 px step, _pool_L5w8 35,771 words md5 ad8748ba); wide12 canvas 1232x364 = base + 12 "
               "candidates (min_gap=1); enumerated body with 12 questions, one per line = 7 present absolute "
               "+ 2 absent absolute (none) + 3 depth-1 chains 'figure k+X' = figure k rotated a further X "
               "(distinct roots, target always present); answer = 12 tokens in 1..12|none; " + _COMMON),
}

_font_checked = False


def _check_font():
    """Pin the badge font."""
    global _font_checked
    if not _font_checked:
        if not FONT.is_file() or hashlib.md5(FONT.read_bytes()).hexdigest() != FONT_MD5:
            raise RuntimeError(f"{FONT}: missing or not the pinned DejaVuSans-Bold (md5 {FONT_MD5})")
        _font_checked = True


def _canon(word):
    k = len(word)
    return min(tuple(word[(i + s) % k] for i in range(k)) for s in range(k))


def _variant_class(word, L):
    k = len(word)
    rev = tuple(word[(-i) % k] for i in range(k))
    seeds = (tuple(word), rev, tuple(L - 1 - c for c in word), tuple(L - 1 - c for c in rev))
    return min(_canon(s) for s in seeds)


def figure_key(word, L) -> str:
    return f"L{L}:" + "".join(map(str, _canon(word)))


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    _check_font()
    if tier not in ladder.TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    r = ladder.build(int(seed), int(index), tier)
    if r is None:
        raise api.Reject("ladder.build: 600 draws without a word clearing the G1/G5 gates")
    spec = ladder.TIERS[tier]
    lay = ladder.LAYOUTS[r["lay"]]
    word = [int(x) for x in r["w"]]
    L = int(r["L"])
    shifts = [int(s) for s in r["shifts"]]
    plan = [dict(p) for p in r["plan"]]
    meta = {
        "generator": "generator/ladder.py:build",
        "rngkey": f"{spec['rngkey']}:{int(seed)}:{int(index)}",
        "L": L,
        "level_step_px": float(r["step"]),
        "pool": r["pool"],
        "pool_md5": wp.SPEC[r["pool"]]["md5"],
        "layout": r["lay"],
        "canvas": [lay["CW"], lay["CH"]],
        "slots": int(r["slots"]),
        "min_gap": int(r["min_gap"]),
        "word": word,                   # base figure: word[j] = level of the spoke at UP + j*22.5 deg clockwise
        "colour": [int(c) for c in r["col"]],
        "shifts": shifts,               # figure i+1 = base rotated clockwise by shifts[i]*22.5 deg
        "shifts_deg": [360.0 * s / 16 for s in shifts],
        "plan": plan,                   # questions in order: abs {step} or chain {ref, steps}; absent -> none
        "n_q": int(r["n_q"]),
        "n_none": int(r["n_none"]),
        "n_chain": int(r["n_chain"]),
        "n_present": int(r["n_present"]),
        "answer_tokens": list(r["ansl"]),
        "figure_key": figure_key(word, L),
        "variant_class": "".join(map(str, _variant_class(word, L))),
        "min_hamming": int(r["minham"]),
        "m_anchor": int(r["m"]),
        "coarse_anchor": int(r["coarse"]),
        "probe_m1": float(r["m1"]),
        "probe_m2": float(r["m2"]),
        "probe_m3": float(r["m3"]),
        "centroid_meas_px": float(r["cent_meas"]),
        "centroid_pred_px": float(r["cent_pred"]),
        "gate_cc": int(r["cc"]),
        "gate_rotdiff": float(r["rotdiff"]),
        "depends_on": int(r["dep"]),
        "tries": int(r["tries"]),
    }
    core_key = digest({"task": TASK, "kind": "base-figure-necklace", "L": L, "necklace": list(_canon(word))})
    return api.Item(question=r["qtext"], answer=r["ans"], image=r["q"], meta=meta, core_key=core_key,
                    answer_type="token-list")
