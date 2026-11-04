"""The `strikes` icon eviction as a pure function.

WHY
---
About a quarter of the icons cannot carry c separable rounded blemishes on a
92 px contour, and with 7-8 base icons per item a pool that keeps them fails
clone synthesis on most seeds.  A sequential search would evict an icon once
its look-alike family fails (once at c >= 4, twice below), but that is
PATH-DEPENDENT: which icons are struck depends on every seed tried so far, so an
item would not be a function of its own seed.  This script computes the steady
state instead, once: for every icon a rung can draw, it builds that icon's
look-alike family alone, with the rung's own parameters, K times with fixed
seeds, and records how many attempts failed.  The adapter drops every icon that
failed at least once and nothing else -- it removes icons, never gates.

    VISRO_FTS_CACHE=<cache> python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_strikes

writes $VISRO_FTS_CACHE/strikes_92.json (most of build_cache's ~20 min on 6 processes;
VISRO_FTS_WORKERS sets the process count).  Deterministic: each trial's RNG is
random.Random("fts-strike/<icon path>/<key>/<trial>"), and a census stops at an
icon's first failure, so the output does not depend on the process count.

KEYS (one per distinct family geometry; avoid_frac depends on the rung's c):
    c1        easy / medium letter and answerless families, c = 1
    hard2     hard, c = 2
    xhard3    xhard, c = 3
    xxhard4   xxhard letter families, c = 4
    xxhard3   xxhard answerless family, 3 clones at xxhard's avoid_frac
A rung strikes the union of its keys.
"""
import os, json, time, random

import numpy as np

import shared.iconlib as iconlib

from .. import icon_pool, puzzle
from ..cache import CACHE_DIR

if not iconlib.HAVE_CAIROSVG:
    raise SystemExit("build_strikes.py needs cairosvg")

K = 4
NPROC = int(os.environ.get("VISRO_FTS_WORKERS", 6))
KEYS = {  # key: (tier whose parameters are used, clones in the family)
    "c1": ("easy", 1),
    "hard2": ("hard", 2),
    "xhard3": ("xhard", 3),
    "xxhard4": ("xxhard", 4),
    "xxhard3": ("xxhard", 3),
}
TIER_KEYS = {"easy": ["c1"], "medium": ["c1"], "hard": ["hard2"], "xhard": ["xhard3"],
             "xxhard": ["xxhard4", "xxhard3"]}

_S = {}


def load():
    """The tables exactly as the adapter loads them, upright rows only."""
    ents = icon_pool.load()[0]
    keep = np.load(CACHE_DIR / "keep3.npy")
    sub = np.load(CACHE_DIR / "sub.npy")
    icons = [int(keep[i]) for i in sub]
    sil = np.array([ents[i]["style"] == "silhouette" for i in icons])
    M = np.load(CACHE_DIR / "Mr_92.npy"); RAD = np.load(CACHE_DIR / "RADr_92.npy")
    OUT = np.load(CACHE_DIR / "OUTr_92.npy"); fill = np.load(CACHE_DIR / "fillr_92.npy")
    from shared.iconlib import IconLibrary
    return dict(ents=ents, icons=icons, sil=sil, M=M, RAD=RAD, OUT=OUT, fill=fill,
                lib=IconLibrary(default_split="train"))


def candidates(tab, tier):
    """Every table row the rung could ever draw from a pristine pool."""
    dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
    avail = (tab["fill"] >= dp["fill_min"]) & tab["sil"]
    if dp["natural"] == "off":
        return [int(i) for i in np.flatnonzero(avail)]
    E = puzzle.select_tables(tab["M"], tab["RAD"], tab["OUT"], avail, dp)[3]
    return sorted({int(x) for x in E.ravel()})


def _one(args):
    key, row = args
    tab = _S["tab"]
    tier, c = KEYS[key]
    dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
    dp.update(L=1, NU=0, letter_clones=[c], u_clones=[])
    path = tab["ents"][tab["icons"][row]]["path"]
    fails = 0
    for trial in range(K):
        rng = random.Random(f"fts-strike/{path}/{key}/{trial}")
        got = puzzle.build_shadows(tab["lib"], tab["ents"], [(tab["icons"][row], False)], dp, rng)
        if got is None:
            fails += 1
            break                      # one failure strikes; stop paying for it
    return key, row, fails, trial + 1


def main():
    from multiprocessing import get_context
    out = CACHE_DIR / "strikes_92.json"
    t0 = time.time()
    tab = load()
    _S["tab"] = tab
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    jobs = []
    rows_of = {}
    for key, (tier, c) in KEYS.items():
        rows_of[key] = candidates(tab, tier)
        jobs += [(key, r) for r in rows_of[key]]
    print(f"{len(jobs)} census jobs over {', '.join(f'{k}:{len(v)}' for k, v in rows_of.items())}",
          flush=True)
    res = {k: {} for k in KEYS}
    with get_context("fork").Pool(NPROC) as p:
        for n, (key, row, fails, trials) in enumerate(p.imap_unordered(_one, jobs, chunksize=2)):
            res[key][row] = [fails, trials]
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    doc = dict(
        format="fts-strikes@1", K=K, rule="strike a row when fails >= 1 for any key of its rung",
        n_rows=len(tab["icons"]), keys={k: dict(tier=t, clones=c) for k, (t, c) in KEYS.items()},
        tier_keys=TIER_KEYS,
        census={k: {str(r): v for r, v in sorted(res[k].items())} for k in KEYS},
    )
    out.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    for k in KEYS:
        n = len(res[k]); s = sum(1 for v in res[k].values() if v[0])
        print(f"{k}: struck {s}/{n} ({s / max(n, 1):.1%})", flush=True)
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
