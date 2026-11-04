"""_pool_L3w8full / _pool_L4w8full: the easy tier's word pools, enumerated exhaustively.

The pools keep W1..W8 and every combinatorial gate.  W5 is not an
anti-shortcut gate and is not negotiable: the minimum pairwise Hamming distance
over the 16 shifts is what keeps the candidates mutually distinguishable.  The
pools do not apply W9 (A_2 = 0, the exact kill of the second ink multipole),
whose family is far too small at these lengths (7 words at L=3 and 40 at L=4
after the other gates).

Construction: EXHAUSTIVE over all L^16 words, no sampling anywhere.

    L=3   3^16 =    43,046,721
    L=4   4^16 = 4,294,967,296

Both are enumerated in full by splitting the word in half.  Because
zeta^(j+8) = -zeta^j,

    A_1(w) = sum_{j<8} q(w_j) zeta^j  -  sum_{j<8} q(w_{j+8}) zeta^j
           = U(a) - U(b),      a = w[0:8],  b = w[8:16]

and the ink mass is likewise additive over the two halves, so W8 over all L^16
words is ONE outer product over the L^8 half-words (6,561 at L=3, 65,536 at
L=4).  Everything that clears W8 is then run through a vectorised transcription
of word.combinatorial_ok (W1..W7).
"""
from __future__ import annotations
import time, math, cmath
import numpy as np

from .. import dial as D, moment as MO
from ..word import K
from . import word_gates as P          # MAX_PRED and CFG
MASK32 = np.int64(0xFFFFFFFF)


# ------------------------------------------------------- 2-bit word packing
def pack(w):
    """(n,16) digits -> one int64 holding 2 bits per digit, MSB = position 0.
    Valid for L <= 4, which is every tier this file builds."""
    c = np.zeros(w.shape[0], dtype=np.int64)
    for j in range(K):
        c = (c << np.int64(2)) | w[:, j].astype(np.int64)
    return c


def unpack(c):
    w = np.empty((c.shape[0], K), dtype=np.uint8)
    for j in range(K):
        w[:, j] = (c >> np.int64(2 * (K - 1 - j))) & np.int64(3)
    return w


def canon_code(c):
    """min over the 16 cyclic rotations - a 2-bit rotation of the packed word."""
    best = c.copy()
    cur = c
    for _ in range(K - 1):
        cur = ((cur << np.int64(2)) | (cur >> np.int64(2 * (K - 1)))) & MASK32
        np.minimum(best, cur, out=best)
    return best


# --------------------------------------------------------------- W8 by halves
def half_tables(L):
    """U(x) and ink mass for every one of the L^8 half-words x = w[0:8]."""
    lv = D.levels(L)
    q = np.array([MO.q_of(p, 1) for p in lv], dtype=np.float64)
    q = q - q.mean()                   # a constant cancels in U(a) - U(b)
    mc = np.array([D.ray_mass(p) for p in lv], dtype=np.float64)
    z = np.array([cmath.exp(2j * math.pi * j / K) for j in range(8)])
    n = L ** 8
    idx = np.arange(n, dtype=np.int64)
    dig = np.empty((n, 8), dtype=np.uint8)
    for j in range(8):                 # digit j is the coefficient of zeta^j
        dig[:, j] = (idx // (L ** j)) % L
    U = (q[dig] * z[None, :]).sum(axis=1)
    return (dig, np.ascontiguousarray(U.real, dtype=np.float32),
            np.ascontiguousarray(U.imag, dtype=np.float32),
            mc[dig].sum(axis=1).astype(np.float32))


def w8_blocks(L, block=64):
    """Yield (words, n_selected) over EVERY word in L^16 that clears W8."""
    dig, Ur, Ui, M = half_tables(L)
    conc = np.float32(math.pi * D.R_HUB ** 2 + 2 * math.pi * D.R_RIM * D.W_RIM)
    thr = np.float32(P.MAX_PRED)
    n = L ** 8
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        dr = Ur[lo:hi, None] - Ur[None, :]
        di = Ui[lo:hi, None] - Ui[None, :]
        d2 = dr * dr + di * di
        lim = thr * (conc + M[lo:hi, None] + M[None, :])
        sel = d2 <= lim * lim
        ai, bi = np.nonzero(sel)
        if ai.size == 0:
            continue
        yield np.concatenate([dig[lo + ai], dig[bi]], axis=1), int(ai.size)


# ------------------------------------------------- W1..W7, vectorised verbatim
def _rot(w, s):
    return np.roll(w, -s, axis=1)


def _no_unique_window(w, L, n):
    """True where NO cyclic window of length n occurs exactly once."""
    c = np.zeros(w.shape, dtype=np.int64)
    for t in range(n):
        c = c * L + _rot(w, t).astype(np.int64)
    cs = np.sort(c, axis=1)
    same_l = np.zeros(cs.shape, dtype=bool)
    same_r = np.zeros(cs.shape, dtype=bool)
    same_l[:, 1:] = cs[:, 1:] == cs[:, :-1]
    same_r[:, :-1] = cs[:, :-1] == cs[:, 1:]
    return ~np.any(~(same_l | same_r), axis=1)


def _both_extremes_repeat(v):
    """word._argext_ok: the max and the min are each attained at least twice."""
    mx = v.max(axis=1, keepdims=True)
    mn = v.min(axis=1, keepdims=True)
    return ((v == mx).sum(axis=1) >= 2) & ((v == mn).sum(axis=1) >= 2)


def w2_mask(w, L):
    """W2: every level present, every level used at least twice."""
    cnt = np.stack([(w == c).sum(axis=1) for c in range(L)], axis=1)
    return cnt.min(axis=1) >= 2


def w4_mask(w, run_cap):
    """W4: longest cyclic run <= run_cap, and if it is >= 3 it occurs twice."""
    eq = w == np.roll(w, 1, axis=1)
    ln = np.ones(w.shape, dtype=np.int16)
    cur = eq.copy()
    for s in range(1, K):
        ln += cur
        if not cur.any():
            break
        cur = cur & np.roll(eq, s, axis=1)
    mx = ln.max(axis=1)
    nmax = (ln == mx[:, None]).sum(axis=1)
    return (mx <= run_cap) & ~((mx >= 3) & (nmax < 2))


def w5_mask(w, min_ham):
    """W5: min pairwise Hamming over the 16 shifts.  ham(rot_i, rot_j) depends
    only on j-i, so the pairwise minimum is the minimum over s = 1..15."""
    mh = np.full(w.shape[0], K, dtype=np.int16)
    for s in range(1, K):
        np.minimum(mh, (w != _rot(w, s)).sum(axis=1).astype(np.int16), out=mh)
    return mh >= min_ham


def gates_vec(w, L, m_req, run_cap, min_ham, coarse_req=3):
    """Vectorised word.combinatorial_ok over a (n,16) uint8 array of words.

    Cheap-and-selective first, and the array is COMPACTED between stages, so a
    block that is 99% rejected costs almost nothing past the first stage."""
    assert m_req == 2 and coarse_req == 3, (m_req, coarse_req)
    n = w.shape[0]
    keep = np.arange(n)

    def stage(m):
        nonlocal w, keep
        w, keep = w[m], keep[m]

    stage(w2_mask(w, L))                       # W2 (and, with it, W3: m_req=2
    stage(_no_unique_window(w, L, 1))          #     is exactly "no level used
    stage(w4_mask(w, run_cap))                 #     exactly once")
    for p in (1, 2, 4, 8):                     # W1 primitive
        stage(np.any(w != _rot(w, p), axis=1))
    stage(w5_mask(w, min_ham))                 # W5
    wi = w.astype(np.int16)                    # W6 gestalt
    stage(_both_extremes_repeat(wi + _rot(wi, 1)))
    wi = w.astype(np.int16)
    stage(_both_extremes_repeat(_rot(wi, 1) - wi))
    wi = w.astype(np.int16)
    stage(_both_extremes_repeat(wi + _rot(wi, 1) + _rot(wi, 2)))
    for th in range(1, L):                     # W7 coarse
        b = (w >= th).astype(np.uint8)
        stage(_no_unique_window(b, 2, 1))
        b = (w >= th).astype(np.uint8)
        stage(_no_unique_window(b, 2, 2))
    out = np.zeros(n, dtype=bool)
    out[keep] = True
    return out


# ------------------------------------------------------------------ enumerate
def enumerate_pool(L, block=64, verbose=True):
    """Every necklace over L^16 that clears W8 and W1..W7.  Exhaustive."""
    c = dict(P.CFG[L])
    c.pop("trials", None)
    c.pop("seed", None)
    c["enumerated"] = True
    c["a2"] = None                     # W9 deliberately NOT applied
    c["coarse_req"] = 3
    t0 = time.time()
    st = dict(tried=L ** K, w8=0, cheap=0)
    codes = []
    nb = 0
    for words, nsel in w8_blocks(L, block):
        st["w8"] += nsel
        m = w2_mask(words, L)
        m[m] &= w4_mask(words[m], c["run_cap"])
        if m.any():
            st["cheap"] += int(m.sum())
            codes.append(canon_code(pack(words[m])).astype(np.int32))
        nb += 1
        if verbose and nb % 256 == 0:
            print(f"    L={L} block {nb}: W8 {st['w8']:,} "
                  f"W2+W4 {st['cheap']:,}  ({time.time()-t0:.0f}s)", flush=True)
    codes = np.unique(np.concatenate(codes)) if codes else \
        np.zeros(0, dtype=np.int32)
    st["neck"] = int(codes.size)
    words = unpack(codes.astype(np.int64))
    g = gates_vec(words, L, c["m_req"], c["run_cap"], c["min_ham"], 3)
    out = sorted(tuple(int(x) for x in r) for r in words[g])
    st["pool"] = len(out)
    st["secs"] = time.time() - t0
    return out, c, st


# ------------------------------------------------------------- SPEC builders
# word_pools.SPEC calls these by name.  Like length5_pools' builders they return
# exactly canonical()'s arguments and are deterministic, so the md5 recorded in
# SPEC pins the result (the cfg each returns is part of the pinned json).
def build_l3_full():
    """_pool_L3w8full.json - easy (1 index in 16).  Every W1..W8 necklace over 3^16.

    Exhaustive, so there is no seed.  ~25 s and ~0.5 GB peak on one core.
    """
    words, cfg, st = enumerate_pool(3, verbose=False)
    cfg = dict(cfg)
    cfg.update(capped=False, n_full=st["pool"],
               note="stride cap lifted; the 512-word sample is _pool_L3w8.json")
    return 3, cfg, P.MAX_PRED, words


def build_l4_full():
    """_pool_L4w8full.json - easy (15 indices in 16).  Every W1..W8 necklace over 4^16.

    Exhaustive, so there is no seed: the same enumeration order, the same
    canonical form, the same sort.  ~190 s and ~4 GB peak on one core.
    """
    words, cfg, st = enumerate_pool(4, verbose=False)
    cfg = dict(cfg)
    cfg.update(capped=False, n_full=st["pool"],
               note="stride cap lifted; the 512-word sample is _pool_L4w8.json")
    return 4, cfg, P.MAX_PRED, words
