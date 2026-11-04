# -*- coding: utf-8 -*-
"""Dumb solvers that read one global statistic of a figure and infer the shift.

m=1  ink centroid bearing              -> pins s outright, if it is measurable
m=2  principal axis of the ink covariance -> pins s mod 8

Both are measured on DELIVERED pixels with the strict saturation ink, so no grey
furniture can dilute them.  Chance for a 12-slot item is 1/12 = 0.083.
"""
from __future__ import annotations
import math
import numpy as np
from . import dial as D
from .word import K

_G = {}


def _grid(n):
    if n not in _G:
        ys, xs = np.mgrid[0:n, 0:n]
        c = (n - 1) / 2.0
        _G[n] = (xs - c, ys - c)
    return _G[n]


def moment_phase(ink, m):
    """arg of  sum_pixels ink * r^m * exp(i*m*theta)  -- the m-th multipole.
    Under a rotation by angle a it multiplies by exp(i*m*a)."""
    x, y = _grid(ink.shape[0])
    z = (x + 1j * y)
    zz = z ** m
    v = (ink * zz).sum()
    return abs(v) / max(ink.sum(), 1e-9), (math.atan2(v.imag, v.real) if abs(v) else 0.0)


def probe_hits_arr(base_arr, cand_arrs, shifts, m=1, ink=D.color_ink):
    """Hit rate of the m-th-multipole solver over the item's own question set,
    computed from tiles that were already rendered."""
    _, a0 = moment_phase(ink(base_arr), m)
    est = {}
    for i, s in enumerate(shifts):
        _, a = moment_phase(ink(cand_arrs[i]), m)
        # rotating the figure by 360 s/k multiplies the m-th multipole by
        # exp(i*m*2pi*s/k), so the phase step per unit shift is m*2pi/k
        step = m * 2 * math.pi / K
        sh = int(round(((a - a0) % (2 * math.pi)) / step)) % (K // math.gcd(m, K))
        est.setdefault(sh, []).append(i)
    hit = 0.0
    per = K // math.gcd(m, K)
    for i, s in enumerate(shifts):
        v = est.get(s % per, [])
        if i in v:
            hit += 1.0 / len(v)
    return hit / len(shifts)
