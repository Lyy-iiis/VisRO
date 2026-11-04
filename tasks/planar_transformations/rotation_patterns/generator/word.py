# -*- coding: utf-8 -*-
"""L-ary cyclic-word kernel.

A figure is a cyclic word w over {0..L-1} of length k=16; w[j] is the LENGTH
LEVEL of the spoke pointing in direction UP + j*22.5 deg.  A candidate is a
cyclic shift, so the task is exactly "identify the shift".

Combinatorial gates
  primitive(w)   <=> the 16 shifts are pairwise different <=> the answer is unique
  m_anchor(w)    = shortest CONTIGUOUS run of spokes that occurs exactly once.
                   Reading fewer than that many consecutive spokes can NEVER pin
                   the phase; some other shift still matches exactly.
  coarse gate    = the same bounds applied to EVERY threshold binarisation of w.
                   This is what defends the multi-level alphabet: a solver that
                   only sees "long vs short" at any cut still has to read >= 3.

Shortcut gates on the ink multipoles - see moment.py.
"""
from __future__ import annotations
import itertools
import collections

K = 16


def rots(w):
    k = len(w)
    return [tuple(w[(i + s) % k] for i in range(k)) for s in range(k)]


def canon(w):
    return min(rots(w))


def primitive(w):
    k = len(w)
    for p in range(1, k):
        if k % p == 0 and all(w[i] == w[(i + p) % k] for i in range(k)):
            return False
    return True


def windows(w, n):
    k = len(w)
    return [tuple(w[(i + t) % k] for t in range(n)) for i in range(k)]


def m_anchor(w):
    for n in range(1, len(w) + 1):
        c = collections.Counter(windows(w, n))
        if any(v == 1 for v in c.values()):
            return n
    return None


def min_hamming(w):
    rr = rots(w)
    return min(sum(a != b for a, b in zip(rr[i], rr[j]))
               for i, j in itertools.combinations(range(len(w)), 2))


def runs(w):
    k = len(w)
    st = next((i for i in range(k) if w[i] != w[(i - 1) % k]), None)
    if st is None:
        return [k]
    out, cur = [], 1
    for t in range(1, k):
        if w[(st + t) % k] == w[(st + t - 1) % k]:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


def thresholds(w, L):
    return [tuple(1 if c >= th else 0 for c in w) for th in range(1, L)]


def rev(w):
    return tuple(w[(-i) % len(w)] for i in range(len(w)))


def flip(w, L):
    return tuple(L - 1 - c for c in w)


def variants(w, L):
    seeds = {tuple(w), rev(w), flip(w, L), flip(rev(w), L)}
    out = set()
    for s in seeds:
        out.update(rots(s))
    return sorted(out)


def _argext_ok(vals):
    """A per-position feature whose max (or min) is attained exactly once is a
    landmark: find it, read its bearing, done.  Both extremes must repeat."""
    mx, mn = max(vals), min(vals)
    return vals.count(mx) >= 2 and vals.count(mn) >= 2


def gestalt_ok(w):
    k = len(w)
    pair = [w[j] + w[(j + 1) % k] for j in range(k)]
    step = [w[(j + 1) % k] - w[j] for j in range(k)]
    tri = [w[j] + w[(j + 1) % k] + w[(j + 2) % k] for j in range(k)]
    return _argext_ok(pair) and _argext_ok(step) and _argext_ok(tri)


def coarse_anchor(w, L):
    """min over all threshold binarisations of m_anchor (99 = no unique window)."""
    out = []
    for b in thresholds(w, L):
        ma = m_anchor(b)
        out.append(99 if ma is None else ma)
    return min(out) if out else 99


def combinatorial_ok(w, L, m_req, run_cap, min_ham, coarse_req=3):
    if not primitive(w):
        return False
    c = collections.Counter(w)
    if len(c) != L or min(c.values()) < 2:
        return False
    ma = m_anchor(w)
    if ma is None or ma < m_req:
        return False
    r = runs(w)
    if max(r) > run_cap:
        return False
    if max(r) >= 3 and r.count(max(r)) < 2:
        return False
    if min_hamming(w) < min_ham:
        return False
    if not gestalt_ok(w):
        return False
    return coarse_anchor(w, L) >= coarse_req
