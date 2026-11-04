"""_pool_L5w8 / _pool_L5xwd / _pool_L5m: the L=5 word pools.

All keep every word gate W1..W8 (W5 included) and draw from EVERY level multiset
that W2 admits (each level used at least twice), not only from the arrangements
of one balanced multiset; that is what gives the upper tiers tens of thousands
of figures.

    _pool_L5w8    hard + xxhard.  W1..W8, every multiset, fixed-seed vectorised
                  random search (below).
    _pool_L5xwd   xhard.  The near-twin family of twin_pools.py with d(8) in
                  {6, 8, 10}, W5's floor tied to each word's own d(8), so that
                  the 180 deg twin is still the closest pair of rings on the
                  canvas.
    _pool_L5m     medium.  hard's W1..W7 family enumerated over all of 5^16,
                  with W8 at MED_MAX_PRED = 0.17 px.

WHY A SEARCH AND NOT AN ENUMERATION (_pool_L5w8)
------------------------------------------------
exhaustive_pools.py enumerates L^16 in full by splitting the word in half
(zeta^(j+8) = -zeta^j makes A_1 = U(a) - U(b) and the ink mass additive).  At
L=5 the two halves are 5^8 = 390,625 each, so the outer product is 1.526e11
pairs; ~1.5% of them clear W8, which is 2.3e9 words to deduplicate and gate.
So _pool_L5w8 is a FIXED-SEED RANDOM SEARCH, run through the same half-word
tables, which makes one trial two table lookups instead of sixteen complex
multiplies.

The search's estimate of the whole family, checked against the two families
whose exact size is known:

    L=3  estimate 1,556   exhaustive_pools.py enumerated 1,553   ratio 1.002
    L=4  estimate 35,098  exhaustive_pools.py enumerated 34,811  ratio 1.008
    L=5  estimate 173,044 necklaces
"""
from __future__ import annotations
import os, time, math, cmath, collections
import numpy as np

from .. import dial as D, moment as MO
from ..word import K
from . import exhaustive_pools as PW, twin_pools as PT, word_gates as P

L5 = 5

# ---- the search's own constants.  Changing ANY of them changes the pool, and
# ---- word_pools.SPEC carries the md5 that says so.
SEED = 20260908
BLOCK = 20_000_000          # words per block
NBLOCK = 110                # blocks -> 2.2e9 trials.  Fixed, not "until full",
                            # so the output is a function of the constants alone.
TWIN_WORKERS = 6            # only affects wall time, never the output


# ------------------------------------------------------- 3-bit packing (L<=8)
def pack5(w):
    c = np.zeros(w.shape[0], dtype=np.int64)
    for j in range(K):
        c = (c << np.int64(3)) | w[:, j].astype(np.int64)
    return c


def unpack5(c):
    w = np.empty((c.shape[0], K), dtype=np.uint8)
    for j in range(K):
        w[:, j] = (c >> np.int64(3 * (K - 1 - j))) & np.int64(7)
    return w


def canon5(c):
    """min over the 16 cyclic rotations of a 3-bit-packed 16-letter word."""
    mask = np.int64((1 << (3 * K)) - 1)
    best = c.copy()
    cur = c
    for _ in range(K - 1):
        cur = ((cur << np.int64(3)) | (cur >> np.int64(3 * (K - 1)))) & mask
        np.minimum(best, cur, out=best)
    return best


# ------------------------------------------------------------- half-word table
def half_tables(L):
    """U(x) and ink mass for every one of the L^8 half-words x = w[0:8].
    Same identity exhaustive_pools.half_tables uses; kept in float64 here because at L=5
    the |A_1| band is ~0.05 px and float32 rounding is a fifth of that."""
    lv = D.levels(L)
    q = np.array([MO.q_of(p, 1) for p in lv], dtype=np.float64)
    mc = np.array([D.ray_mass(p) for p in lv], dtype=np.float64)
    z = np.array([cmath.exp(2j * math.pi * j / K) for j in range(8)])
    n = L ** 8
    idx = np.arange(n, dtype=np.int64)
    dig = np.empty((n, 8), dtype=np.uint8)
    for j in range(8):
        dig[:, j] = (idx // (L ** j)) % L
    U = (q[dig] * z[None, :]).sum(axis=1)
    return (dig, np.ascontiguousarray(U.real), np.ascontiguousarray(U.imag),
            mc[dig].sum(axis=1))


def conc_mass():
    return math.pi * D.R_HUB ** 2 + 2 * math.pi * D.R_RIM * D.W_RIM


# ------------------------------------------------------------------ the search
def search(L, nblock, block, seed, min_ham=None, verbose=True, thr=None):
    """Fixed-seed uniform random search over ALL of L^16.

    Block b draws from default_rng([seed, L, b]) and the blocks are consumed in
    order, so the result is a function of (seed, L, nblock, block) alone - the
    same on any machine, with or without parallelism.

    min_ham=None uses word_gates.CFG[L]['min_ham'].  Passing 0 turns W5
    off and nothing else."""
    dig, Ur, Ui, M = half_tables(L)
    conc = conc_mass()
    c = P.CFG[L]
    mh = c["min_ham"] if min_ham is None else min_ham
    thr = P.MAX_PRED if thr is None else thr
    n = L ** 8
    seen = set()
    st = dict(tried=0, w8=0, hits=0)
    t0 = time.time()
    for b in range(nblock):
        rng = np.random.default_rng([seed, L, b])
        a = rng.integers(0, n, size=block, dtype=np.int64)
        bb = rng.integers(0, n, size=block, dtype=np.int64)
        dr = Ur[a] - Ur[bb]
        di = Ui[a] - Ui[bb]
        lim = thr * (conc + M[a] + M[bb])
        sel = dr * dr + di * di <= lim * lim
        st["tried"] += block
        st["w8"] += int(sel.sum())
        if sel.any():
            wds = np.concatenate([dig[a[sel]], dig[bb[sel]]], axis=1)
            g = PW.gates_vec(wds, L, c["m_req"], c["run_cap"], mh, 3)
            if g.any():
                st["hits"] += int(g.sum())
                seen.update(int(x) for x in canon5(pack5(wds[g])))
        if verbose and (b + 1) % 10 == 0:
            print(f"    L={L} block {b+1}/{nblock}: tried {st['tried']:,} "
                  f"W8 {st['w8']:,} hits {st['hits']:,} distinct {len(seen):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    st["secs"] = time.time() - t0
    st["dens"] = st["hits"] / max(st["tried"], 1)
    st["est_words"] = st["dens"] * L ** K
    st["est_neck"] = st["est_words"] / 16
    st["distinct"] = len(seen)
    codes = np.array(sorted(seen), dtype=np.int64)
    words = unpack5(codes) if codes.size else np.zeros((0, K), np.uint8)
    return [tuple(int(x) for x in r) for r in words], st


# ------------------------------------------ the widened near-twin family (xhard)
# d(8) is drawn from XW_D8 instead of being pinned to 6; everything the reader
# sees is unchanged (same wide12 canvas, same 12 candidates, same 10 absolute +
# 2 none questions, same L=5 and its 8.25 px bead step, same min_gap=1 bank).
#
#   relaxed   d(8) = 6 exactly            ->  d(8) in XW_D8
#   kept      W5 = the twin IS the closest pair: min_ham is set to d(8) for
#             every word, so no two rings on the canvas are ever closer than
#             the 180 deg pair (at d(8) = 8 and 10 this is stricter than
#             hard's min_ham 7).
#   kept      W1..W4, W6, W7, W8 and P2.
#
# Tying min_ham to d(8) also stops the family growing with d(8): "exactly 2M of
# the 8 opposite pairs disagree" gets commoner as M rises, but "all 15 shifts
# differ in at least 2M spokes" gets rarer faster.  Per-d(8) family sizes:
#
#     d(8)                        6        8       10       12
#     necklaces, thr 0.20     6,787   12,243   18,993    7,093
#     necklaces, thr 0.10     1,870    3,201    5,600    1,773
#
# No single d(8) reaches 25,000 figures; the union of 6, 8 and 10 does (38,023
# at thr 0.20).  The estimator is a uniform sample of the d(8)=2M subspace; on
# the d(8)=6 / thr 0.08 corner it predicts 1,388 necklaces against the 1,396 of
# the exact enumeration.
XW_D8 = (6, 8, 10)        # the twin is 6, 8 or 10 spokes away
XW_MAX_PRED = P.MAX_PRED  # 0.20, the W8 threshold of every other pool: with 4
                          # and 5 disagreeing pairs |A_1| can cancel below it
XW_SEED = 20260908
XW_BLOCK = 5_000_000      # words per block; the block is also the memory unit
XW_NBLOCK = {6: 12, 8: 52, 10: 168}     # trials = NBLOCK * BLOCK per d(8):
                          # 16*trials/|subspace| = 0.69 / 0.59 / 0.60, about
                          # 45-50% of each family's necklaces.  That suffices
                          # because build() draws word.variants(w0, L), and a
                          # figure is reached as soon as ANY of the ~3 family
                          # words whose variant set holds it is sampled: about
                          # 83% of the family's figures, ~32,000.
XW_WORKERS = 6            # wall time only; the output is per-block and sorted
XW_SCAN = 60000           # P2 render budget, >= the family so nothing is cut


def _twin_sample(M, n, rng):
    """Uniform over {w in 5^16 : exactly M of the 8 opposite pairs disagree}.

    The set is closed under rotation (rot by s sends opposite pair j to pair
    j+s, and rot by 8 swaps the two members of every pair), so sampling it and
    dividing the hit count by 16 estimates NECKLACES directly."""
    w = np.empty((n, K), dtype=np.uint8)
    js = np.argsort(rng.random((n, 8)), axis=1)[:, :M]   # which pairs disagree
    lv = rng.integers(0, L5, size=(n, 8))
    w[:, :8] = lv
    w[:, 8:] = lv
    a = rng.integers(0, L5, size=(n, M))
    b = rng.integers(0, L5 - 1, size=(n, M))
    b = b + (b >= a)                                     # b != a, uniform
    rows = np.arange(n)[:, None]
    w[rows, js] = a
    w[rows, js + 8] = b
    return w


def _twin_block(arg):
    """One block of the search.  Block (M, b) draws from
    default_rng([XW_SEED, M, b]) and nothing else, so the union over blocks is
    a function of the constants alone - the same with 1 worker or with 12."""
    M, b, block, thr = arg
    rng = np.random.default_rng([XW_SEED, M, b])
    w = _twin_sample(M, block, rng)
    lv = D.levels(L5)
    q = np.array([MO.q_of(p, 1) for p in lv])
    rm = np.array([D.ray_mass(p) for p in lv])
    Z = np.array([cmath.exp(2j * math.pi * j / K) for j in range(K)])
    A = (q[w] * Z[None, :]).sum(axis=1)
    sel = np.abs(A) <= thr * (conc_mass() + rm[w].sum(axis=1))
    if not sel.any():
        return []
    ww = w[sel]
    g = PW.gates_vec(ww, L5, PT.CFG["m_req"], PT.CFG["run_cap"], 2 * M,
                     PT.CFG["coarse_req"])
    if not g.any():
        return []
    return sorted({int(x) for x in canon5(pack5(ww[g]))})


def twin_wide_family(d8s=None, thr=None, workers=XW_WORKERS, verbose=True):
    """The union of the d(8) families in XW_D8, each with W5 tied to its own
    d(8).  Fixed-seed random search over each d(8) subspace, not an enumeration:
    the subspaces run to 4.5e10 words and only ~2e-5 of them clear the stack."""
    import multiprocessing as mp
    d8s = XW_D8 if d8s is None else d8s
    thr = XW_MAX_PRED if thr is None else thr
    jobs = [(m // 2, b, XW_BLOCK, thr) for m in d8s
            for b in range(XW_NBLOCK[m])]
    t0 = time.time()
    seen, st = set(), collections.Counter()
    if workers > 1:
        with mp.Pool(workers) as pool:
            for (M, b, _, _), got in zip(jobs, pool.imap(_twin_block, jobs,
                                                         chunksize=1)):
                seen.update(got)
                st[2 * M] = len(got) + st[2 * M]
                if verbose and (b + 1) % 40 == 0:
                    print(f"    d(8)={2*M} block {b+1}/{XW_NBLOCK[2*M]} "
                          f"distinct so far {len(seen):,} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    else:
        for j in jobs:
            seen.update(_twin_block(j))
    codes = np.array(sorted(seen), dtype=np.int64)
    words = unpack5(codes) if codes.size else np.zeros((0, K), np.uint8)
    out = [tuple(int(x) for x in r) for r in words]
    stt = dict(secs=time.time() - t0, uniq=len(out),
               trials=sum(XW_NBLOCK[m] for m in d8s) * XW_BLOCK)
    return out, stt


def build_twin_wide():
    """_pool_L5xwd.json - xhard, widened.  Returns word_pools.canonical's args."""
    words, st = twin_wide_family(verbose=False)
    keep, drop, scanned = twin_p2(words, n_scan=XW_SCAN, workers=XW_WORKERS,
                                  verbose=False)
    d8 = collections.Counter(
        sum(a != b for a, b in zip(w, [w[(j - 8) % K] for j in range(K)]))
        for w in keep)
    cfg = dict(PT.CFG)
    cfg.update(family=f"d(8) in {list(XW_D8)}, min_ham = d(8)",
               d8=list(XW_D8), min_ham=min(XW_D8), enumerated=False,
               search="uniform-random-over-{d(8)=2M}", seed=XW_SEED,
               block=XW_BLOCK, nblock=dict(XW_NBLOCK), trials=st["trials"],
               multiset="unrestricted (W2 only)", min_cc=PT.MIN_CC,
               family_before_p2=st["uniq"], p2_scanned=scanned,
               p2_dropped=drop, d8_hist=dict(sorted(d8.items())))
    return L5, cfg, XW_MAX_PRED, keep


def _p2_one(w):
    """min over the palette of the 180 deg twin's largest connected change."""
    m = 10 ** 9
    for col in D.PALETTE:
        m = min(m, PT.twin_blob(w, col))
        if m < PT.MIN_CC:
            break                      # a fail at any colour is a fail
    return m


def twin_p2(words, min_cc=None, n_scan=None, workers=TWIN_WORKERS,
            verbose=True):
    """P2, unchanged from twin_pools.py: the largest connected change between a
    figure and its 180 deg twin must clear the G1 floor AT EVERY palette colour.

    It costs 2 renders per word per colour (~0.5 s a word), so the family is
    visited in a FIXED order - the sorted list, strided down to n_scan - and the
    scan is spread over `workers` processes.  Determinism does not depend on the
    worker count: the scan set is a function of (len(words), n_scan) and the
    survivors are re-sorted."""
    import multiprocessing as mp
    mc = PT.MIN_CC if min_cc is None else min_cc
    n = len(words)
    ns = n if n_scan is None else min(n, n_scan)
    order = [i * n // ns for i in range(ns)]
    cand = [words[i] for i in order]
    t0 = time.time()
    if workers > 1 and ns > 32:
        with mp.Pool(workers) as pool:
            vals = pool.map(_p2_one, cand, chunksize=16)
    else:
        vals = [_p2_one(w) for w in cand]
    keep = [w for w, v in zip(cand, vals) if v >= mc]
    if verbose:
        print(f"    P2 scanned {ns}/{n} of the family in {time.time()-t0:.0f}s "
              f"on {workers} workers: kept {len(keep)}, dropped {ns-len(keep)}")
    return sorted(keep), ns - len(keep), ns


# ------------------------------------------------ builders called by word_pools.SPEC
# word_pools.SPEC calls these BY NAME and with NO ARGUMENTS: every constant a
# pool depends on is a module constant above, so the md5 in SPEC is a checksum
# of this file's constants as much as of the words.

def build_w8():
    """_pool_L5w8.json - hard and xxhard.  Returns word_pools.canonical's args."""
    words, st = search(L5, NBLOCK, BLOCK, SEED)
    cfg = dict(P.CFG[L5])
    cfg.pop("trials", None)
    cfg.pop("seed", None)
    cfg.update(a2=None, coarse_req=3, enumerated=False,
               search="uniform-random-over-L^16", multiset="unrestricted (W2 only)",
               seed=SEED, nblock=NBLOCK, block=BLOCK, trials=st["tried"],
               hits=st["hits"], density=st["dens"],
               est_necklaces=round(st["est_neck"]))
    return L5, cfg, P.MAX_PRED, words


# --------------------------------------------- medium at L=5
# medium draws from hard's own L=5 word family, ENUMERATED (the law of the L=3 /
# L=4 pools: every variant class enters with every member that clears the
# gates), with ONE gate set for this tier: W8's MAX_PRED is MED_MAX_PRED = 0.17
# instead of 0.20.  W1..W7 are word_gates.CFG[5].  (The L=4 family is too small
# for the task-wide figure key once easy draws from it.)
#
#   * The enumeration is cheap: W2 puts a 0 in every word, so a canonical
#     necklace (the minimum rotation) starts with 0, only half-words a = w[0:8]
#     with a[0] == 0 need pairing (3.05e10 pairs instead of 1.5e11), and the
#     O(k^2) gate stack runs only on pairs that are their own canonical
#     rotation.  ~90 s on 7 cores, ~10 min on one.
#   * At MAX_PRED 0.20 that is the whole family: 173,622 necklaces, 62,535
#     classes, 245,033 figures; hard's _pool_L5w8 is a 20.6% sample of it.
#   * 0.17 lowers the centroid probe's hit rate on medium's 6-slot bank to the
#     level of the neighbouring tiers; it is the one difficulty knob turned, and
#     it is turned towards HARDER.
#   * No class is excluded.  medium and hard / xhard / xxhard share the family;
#     the task-wide figure key keeps every figure unique across tiers.  126,936
#     words, 186,857 figures.
MED_MAX_PRED = 0.17
MED_WORKERS = int(os.environ.get("VISRO_ROTATION_WORKERS", "1"))   # wall time only, never the output
_MED_NCHUNK = 400
_MT = None


def _med_init():
    global _MT
    _MT = half_tables(L5)


def _med_chunk(ch):
    """Every canonical necklace w of the family with w[0:8] among chunk ch of the half-words starting with 0
    (W8 at word_gates.MAX_PRED through the half tables, exactly as search(); then gates_vec)."""
    dig, Ur, Ui, M = _MT
    c = P.CFG[L5]
    conc = conc_mass()
    al = np.nonzero(dig[:, 0] == 0)[0][ch::_MED_NCHUNK]
    out = []
    for i in range(0, al.size, 8):
        aa = al[i:i + 8]
        dr = Ur[aa][:, None] - Ur[None, :]
        di = Ui[aa][:, None] - Ui[None, :]
        lim = P.MAX_PRED * (conc + M[aa][:, None] + M[None, :])
        ia, ib = np.nonzero(dr * dr + di * di <= lim * lim)
        if not ia.size:
            continue
        wds = np.concatenate([dig[aa[ia]], dig[ib]], axis=1)
        codes = pack5(wds)
        can = canon5(codes) == codes
        g = PW.gates_vec(wds[can], L5, c["m_req"], c["run_cap"], c["min_ham"], 3)
        out.append(codes[can][g])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def family_l5(verbose=True):
    """The EXHAUSTIVE L=5 W1..W8 family at word_gates.CFG[5] / MAX_PRED, as sorted canon5 codes."""
    t0 = time.time()
    if MED_WORKERS > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(MED_WORKERS, initializer=_med_init) as pool:
            found = pool.map(_med_chunk, range(_MED_NCHUNK))
    else:
        _med_init()
        found = []
        for ch in range(_MED_NCHUNK):
            found.append(_med_chunk(ch))
            if verbose and (ch + 1) % 50 == 0:
                print(f"    L=5 family chunk {ch+1}/{_MED_NCHUNK} ({time.time()-t0:.0f}s)", flush=True)
    return np.unique(np.concatenate(found))


def build_medium():
    """_pool_L5m.json - medium (rotation_patterns).  Returns word_pools.canonical's args."""
    fam = family_l5(verbose=False)
    words = [tuple(int(x) for x in r) for r in unpack5(fam)]
    words = [w for w in words if MO.centroid_px(w, L5) <= MED_MAX_PRED]
    cfg = dict(P.CFG[L5])
    cfg.pop("trials", None)
    cfg.pop("seed", None)
    cfg.update(a2=None, coarse_req=3, enumerated=True, multiset="unrestricted (W2 only)",
               family_necklaces=int(fam.size), family_max_pred=P.MAX_PRED,
               rule="every necklace of the exhaustive L=5 W1..W8 family (pool_gates.CFG[5], MAX_PRED 0.20) whose "
                    "ideal centroid is <= MED_MAX_PRED; W8 tightened for medium only")
    return L5, cfg, MED_MAX_PRED, words
