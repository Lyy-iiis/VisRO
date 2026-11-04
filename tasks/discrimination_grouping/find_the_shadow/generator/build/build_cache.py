"""Build every cache table the find_the_shadow adapter needs, from nothing but the icon library.

Five stages, each skipped if its output is already on disk:

  pool     pool3.pkl        14,000 icons rendered at 256 px, silhouettes cached
                            as 160x160 packed bitmaps
  keep     keep3.npy        picture-quality filter over the pool
           white3.npy       per-icon "invisible on white paper" score
  rok      rok.npy rfill.npy ragree.npy rmask160.npy
                            re-renders every kept icon through the DELIVERY path
                            (canon_tile(render_canon(...,504))) and records how far
                            that mask is from the cached one
  retab    sub.npy Mr_<T>.npy OUTr_<T>.npy RADr_<T>.npy fillr_<T>.npy
                            the gate tables, tabulated on the DELIVERED mask at the
                            real tile size
  strikes  strikes_92.json  build_strikes.py's census

Usage
    VISRO_FTS_CACHE=<dir> python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_cache
    VISRO_FTS_CACHE=<dir> python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_cache rok retab
    VISRO_FTS_TILES=92,100 ... build_cache retab     # extra tile sizes

Stages pool / keep / rok run their per-icon loops on a fork pool of
VISRO_FTS_WORKERS processes (default 6) and put the results back in the
original iteration order before the "first n accepted" cut.  All five stages
take about 20 min on 6 processes, most of it the strike census.  Only tile 92 is
needed (all rungs use a 92 px tile).

white3.npy is the fraction of an icon's opaque pixels that are near-white, i.e.
the part of a colour icon that is invisible against the white page but solid
black in its shadow; the gate is `white <= 0.10`, and only colour icons ever
score above zero.

The .npy tables come out identical under numpy 1.26 / opencv 4.10 / cairosvg
2.9.0 and numpy 2.5.4 / opencv 5.0 / cairosvg 2.9.1 (pool3.pkl differs only in
how numpy pickles arrays).
"""
import os
import pickle
import random
import sys
import time
from multiprocessing import get_context

import numpy as np
import cv2

from .. import decoys, icon_pool, puzzle
from ..base import silhouette
from ..cache import CACHE_DIR
from . import build_strikes

WHITE_S, WHITE_THR, ALPHA_THR = 160, 240, 40
TILES = [int(x) for x in os.environ.get("VISRO_FTS_TILES", "92").split(",")]
NPROC = int(os.environ.get("VISRO_FTS_WORKERS", 6))

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        from shared.iconlib import IconLibrary
        _LIB = IconLibrary(default_split="train")
    return _LIB


def _pool():
    return get_context("fork").Pool(NPROC)


# ------------------------------------------------------------------ 1. pool
def _pool_one(ii):
    e = lib().icons[ii]
    RENDER_S, STORE_S = icon_pool.RENDER_S, icon_pool.STORE_S
    try:
        img = lib().render(e, RENDER_S,
                           color=(0, 0, 0) if e["style"] == "silhouette" else None)
        full = silhouette.alpha_mask(img)
        m = silhouette.tile_silhouette(img, RENDER_S)
    except Exception:
        return None
    if m.sum() < 400 or full.sum() == 0:
        return None
    nm = decoys.norm_mask(m, STORE_S)
    if nm.sum() < 200:
        return None
    return (ii, float(m.sum()) / float(full.sum()), float(nm.mean()),
            np.packbits(nm.ravel()))


def par_pool_build(n, seed):
    idxs = list(lib()._pool[(None, "train")])
    random.Random(seed).shuffle(idxs)
    with _pool() as p:
        res = p.map(_pool_one, idxs, chunksize=8)
    entries, packed, ratios, fills = [], [], [], []
    for r in res:
        if len(entries) >= n:
            break
        if r is None:
            continue
        ii, ra, fi, pk = r
        entries.append(lib().icons[ii]); ratios.append(ra); fills.append(fi)
        packed.append(pk)
    return (entries, np.stack(packed), np.asarray(ratios), np.asarray(fills))


def stage_pool():
    if icon_pool.POOL_FILE.exists():
        print("pool3.pkl exists, skip"); return
    t0 = time.time()
    out = par_pool_build(icon_pool.POOL_N, icon_pool.POOL_SEED)
    with open(icon_pool.POOL_FILE, "wb") as f:
        pickle.dump(out, f)
    print("pool %d icons %.0fs" % (len(out[0]), time.time() - t0), flush=True)


# ------------------------------------------------------------------ 2. keep
_ENTS = None
_M160 = None


def _shape_one(i):
    m = _M160[i]
    ys, xs = np.where(m)
    bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    a = m.sum()
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    c = max(c, key=cv2.contourArea)
    return a / bb, a / max(cv2.contourArea(cv2.convexHull(c)), 1)


def _white_one(k):
    e = _ENTS[int(k)]
    try:
        img = lib().render(e, WHITE_S,
                           color=(0, 0, 0) if e["style"] == "silhouette" else None)
    except Exception:
        return 1.0
    a = np.asarray(img)
    m = a[..., 3] > ALPHA_THR
    w = (a[..., :3].astype(int) >= WHITE_THR).all(-1) & m
    return float(w.sum()) / max(int(m.sum()), 1)


def stage_keep():
    global _ENTS, _M160
    if (CACHE_DIR / "keep3.npy").exists() and (CACHE_DIR / "white3.npy").exists():
        print("keep3/white3 exist, skip"); return
    ents, packed, R, fill = icon_pool.load()
    _ENTS = ents
    _M160 = icon_pool.unpack(packed)
    n = len(_M160)
    t0 = time.time()
    with _pool() as p:
        res = p.map(_shape_one, range(n), chunksize=32)
    bfl = np.array([r[0] for r in res], float)
    sol = np.array([r[1] for r in res], float)
    # base picture-quality filter + kill CONVEX FILLED SLABS: an icon that is
    # both >=85% of its bbox and >=95% of its convex hull reads as a featureless
    # dark rectangle, and a puzzle of those is a "spot the rim" task.
    keep = np.where((R >= 0.90) & (fill >= 0.12) & (fill <= 0.68)
                    & ~((bfl >= 0.85) & (sol >= 0.95)))[0]
    np.save(CACHE_DIR / "keep3.npy", keep)
    print("kept", len(keep), "%.0fs" % (time.time() - t0), flush=True)
    t0 = time.time()
    with _pool() as p:
        white = np.array(p.map(_white_one, list(keep), chunksize=8))
    np.save(CACHE_DIR / "white3.npy", white)
    print("white>0.10:", int((white > 0.10).sum()), "/", len(keep),
          "%.0fs" % (time.time() - t0), flush=True)


# ------------------------------------------------------------------- 3. rok
_KEEP = None


def _rok_one(i):
    k = int(_KEEP[i])
    out = (0.0, 0.0, 0.0, 0.0, np.zeros(160 * 160 // 8, np.uint8))
    try:
        img = puzzle.canon_tile(silhouette.render_canon(lib(), _ENTS[k], 504,
                                                        (0, 0, 0)), 504)
        m = np.asarray(img)[..., 3] > 40
    except Exception:
        return out
    if m.sum() < 500:
        return out
    nc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    big = st[1:, cv2.CC_STAT_AREA].max() / max(m.sum(), 1) if nc > 1 else 1.0
    n126 = decoys.norm_mask(m, 126); n102 = decoys.norm_mask(m, 102)
    ag126 = decoys.iou(n126, decoys.norm_mask(_M160[i], 126))
    ag102 = decoys.iou(n102, decoys.norm_mask(_M160[i], 102))
    fil = float(decoys.norm_mask(m, 160).mean())
    store = np.packbits(decoys.norm_mask(m, 160).ravel())
    return (big, ag126, ag102, fil, store)


def stage_rok():
    """Per-icon compatibility of the DELIVERY render path with the cached masks.

    The cache tabulates `tile_silhouette` masks (largest connected component of a
    256 px render).  The shipped image draws `canon_tile(render_canon(...))` and
    the shadow is built from ITS alpha.  For a quarter of the pool those are not
    the same mask, which makes every cache-based gate a fiction for that icon and
    makes the clone synthesiser fail outright (it refuses to change topology).
    Both are fixed by using only icons where the two agree.
    """
    global _ENTS, _M160, _KEEP
    if (CACHE_DIR / "rmask160.npy").exists():
        print("rok tables exist, skip"); return
    ents, packed, R, fill = icon_pool.load()
    keep = np.load(CACHE_DIR / "keep3.npy")
    _ENTS, _KEEP = ents, keep
    _M160 = icon_pool.unpack(packed, keep)
    n = len(keep)
    t0 = time.time()
    with _pool() as p:
        res = p.map(_rok_one, range(n), chunksize=8)
    big = np.zeros(n, np.float32); ag126 = np.zeros(n, np.float32)
    ag102 = np.zeros(n, np.float32); fil = np.zeros(n, np.float32)
    store = np.zeros((n, 160 * 160 // 8), np.uint8)
    for i, r in enumerate(res):
        big[i], ag126[i], ag102[i], fil[i], store[i] = r
    ok = (big >= 0.998) & (ag126 >= 0.985) & (ag102 >= 0.982) & (fil > 0)
    np.save(CACHE_DIR / "rok.npy", ok); np.save(CACHE_DIR / "rfill.npy", fil)
    np.save(CACHE_DIR / "ragree.npy", np.stack([big, ag126, ag102]))
    np.save(CACHE_DIR / "rmask160.npy", store)
    print("render_ok", int(ok.sum()), "/", n, "%.0fs" % (time.time() - t0),
          flush=True)


# ----------------------------------------------------------------- 4. retab
def stage_retab():
    """The gate tables, built on the masks the SHIPPED image actually draws.

    Recomputing them at the real tile size (instead of reusing the 256 px
    tile_silhouette cache, which differs by ~0.03 IoU for most of the pool and is
    a different mask entirely for a quarter of it) makes every gate exact rather
    than conservative.
    """
    white = np.load(CACHE_DIR / "white3.npy")
    ag = np.load(CACHE_DIR / "ragree.npy"); fil = np.load(CACHE_DIR / "rfill.npy")
    store = np.load(CACHE_DIR / "rmask160.npy")
    big = ag[0]
    sub = np.flatnonzero((big >= 0.998) & (fil >= 0.24) & (white <= 0.10))
    print("subset", len(sub), flush=True)
    np.save(CACHE_DIR / "sub.npy", sub)
    m160 = np.unpackbits(store[sub], axis=1)[:, :160 * 160].reshape(
        -1, 160, 160).astype(bool)

    for T in TILES:
        if (CACHE_DIR / f"RADr_{T}.npy").exists():
            print("tile", T, "exists, skip"); continue
        t0 = time.time()
        NM = np.stack([decoys.norm_mask(m, T) for m in m160])
        F = NM.reshape(len(sub), -1).astype(np.float32)
        area = F.sum(1)
        inter = F @ F.T
        M = (inter / np.maximum(area[:, None] + area[None, :] - inter, 1.0)
             ).astype(np.float32)
        np.fill_diagonal(M, 1.0)
        OF = np.stack([decoys.outline_fill(m) for m in NM]).reshape(
            len(sub), -1).astype(np.float32)
        oa = OF.sum(1); oi = OF @ OF.T
        OUT = (oi / np.maximum(oa[:, None] + oa[None, :] - oi, 1.0)
               ).astype(np.float32)
        np.fill_diagonal(OUT, 1.0)
        print(T, "M+OUT %.0fs" % (time.time() - t0), flush=True)
        P = np.argwhere(np.triu(M >= 0.62, 1))
        RAD = np.zeros_like(M)
        t1 = time.time()
        for a, b in P:
            x = np.logical_xor(NM[a], NM[b]).astype(np.uint8)
            RAD[a, b] = RAD[b, a] = cv2.distanceTransform(x, cv2.DIST_L2, 5).max()
        np.save(CACHE_DIR / f"Mr_{T}.npy", M); np.save(CACHE_DIR / f"OUTr_{T}.npy", OUT)
        np.save(CACHE_DIR / f"RADr_{T}.npy", RAD)
        np.save(CACHE_DIR / f"fillr_{T}.npy",
                (NM.reshape(len(sub), -1).mean(1)).astype(np.float32))
        print(T, "pairs", len(P), "RAD %.0fs" % (time.time() - t1), flush=True)


# --------------------------------------------------------------- 5. strikes
def stage_strikes():
    if (CACHE_DIR / "strikes_92.json").exists():
        print("strikes_92.json exists, skip"); return
    build_strikes.main()


STAGES = dict(pool=stage_pool, keep=stage_keep, rok=stage_rok, retab=stage_retab, strikes=stage_strikes)

if __name__ == "__main__":
    if "VISRO_FTS_CACHE" not in os.environ:
        raise SystemExit("set VISRO_FTS_CACHE to the cache directory")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or list(STAGES)
    print("VISRO_FTS_CACHE =", CACHE_DIR, "NPROC", NPROC, flush=True)
    for s in want:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
        t0 = time.time()
        print("== stage", s, "==", flush=True)
        STAGES[s]()
        print("== stage", s, "done %.0fs" % (time.time() - t0), flush=True)
