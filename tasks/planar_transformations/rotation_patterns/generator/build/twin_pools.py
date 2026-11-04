"""The near-twin word family: words whose 180 deg twin is close (d(8) small).
length5_pools.py builds xhard's pool from it.

Every figure of an item is a cyclic shift of the same 16-letter word w, so the
13 rings on the canvas (base + 12) form a cyclic code, and how confusable two of
them are is one number:

    d(s) = #{ j : w[j] != w[j-s] }     rings s apart differ in d(s) of 16 spokes

For a word of the other pools, d(s) sits near 13.6 for every s (the average is
FIXED at 204/15 = 13.6 by the level multiset alone): any two figures on the
canvas share barely 2 of their 16 spokes, and the reader can tell them apart
from almost any three beads.

This family instead pins the ONE shift that every question walks through:

    P1   d(8) = 6

so a figure and the figure 180 deg from it agree on 10 of their 16 spokes
instead of on 2.4.  Because the bank is 12 of the 14 legal shifts, at least 10
of the 12 questions have their 180 deg partner ON THE CANVAS: for almost every
question there is a decoy figure that only six beads separate from the right
answer.  It also drops the code's minimum distance from ~10 to 6, i.e. its
error-correction radius from 4 misread spokes to 2.

This cannot be relativised away: d(8) is a property of the WORD, so it governs
every pair of rings 180 deg apart alike, and every question of the tier names
the base, so the base's ring is one of the two rings in every alignment.

Why not d(8) = 4: A_1 of a word with d(8) = 2m is the sum of just m terms (the
agreeing opposite pairs cancel exactly), so with m = 2 no word gets the ink
centroid offset under the W8 gate.  m = 3 has three terms, cancels to 0.027 px
and clears it.

  P2  the 6 beads that separate a figure from its 180 deg twin must survive the
      delivered raster: the largest connected change between the two tiles must
      clear the SAME G1 floor of 110 px^2 the other tiers clear, at EVERY colour
      in the palette.

The other word gates are word_gates.py's: W1 primitive, W2 every level used
>= 2, W3 m_anchor >= 2, W4 runs <= 3 and the longest run occurs twice, W6
gestalt, W7 coarse anchor >= 3, W8 ideal ink-centroid offset <= MAX_PRED
delivered px.

Construction is an exact enumeration, not a search.  d(8) = 2m means exactly m
of the eight opposite pairs (j, j+8) disagree; A_1 depends only on those m pairs,
so the centroid gate is evaluated on the (positions, level pairs) skeleton and
only the survivors are expanded into the arrangements of the agreeing pairs.
Every level multiset with each level used at least twice is covered.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import dial as D
from ..word import K

L = 5
M = 3                     # d(8) = 2M = 6
MAX_PRED = 0.08           # W8 bound of this family (it reaches 0.021)
MIN_CC = 110              # the same G1 floor the other tiers clear
CFG = dict(m_req=2, run_cap=3, min_ham=2 * M, coarse_req=3, d8=2 * M,
           family=f"d(8)={2*M}")


def twin_blob(w, col):
    """Largest connected change between a figure and its 180 deg twin, on
    delivered pixels, with the same threshold G1 uses."""
    a = D.flatten(D.render_tile(tuple(w), L, col))
    b = D.flatten(D.render_tile(tuple(w[(j - 8) % K] for j in range(K)), L, col))
    dm = np.abs(a - b).max(axis=2) > 45
    lab, n = ndimage.label(dm)
    return int(ndimage.sum(dm, lab, range(1, n + 1)).max()) if n else 0
