"""rotation_patterns geometry gates: the dial is L = HOW MANY SPOKE LENGTHS.

The symbol is one SOLID spoke per direction, running outward from the central
disc, ending in a round bead.  How far out the bead sits IS the code.  No break,
no second piece, no mirror.

The radial ladder keeps the SAME two rails at every L (bead centre at 34 px and
at 67 px from the middle, on delivered pixels), so a larger L only inserts rungs
between the fixed rails and shrinks the step: 16.5 / 11 / 8.25 delivered px at
L = 3 / 4 / 5.  There are k = 16 spokes on a 168 px figure box.

============================================== the ink-centroid shortcut
A candidate IS the base rotated, so the bearing of its ink centroid rotates with
it and reading that bearing hands over the shift with no reading of the code at
all.  It is closed by FILTERING THE WORD: moment.py shows exact cancellation is
impossible for a plain length code, so the pools keep only the words that sit
on the |A_1| floor (W8).  The centroid probe itself (answer_probe.py) is recorded in
every item's meta, not gated.

gates
 S1  slot shuffle + question shuffle
 S2  bank excludes 0 deg and 180 deg and is a proper subset of the 14 shifts
 S4  exact 16-fold angular lattice, one spoke straight up -> bearing entropy 0
 G1  worst_cc >= MIN_CC        any two delivered figures differ by a real blob
 G3  complete solver: every question matched exhaustively against all
     candidates as cyclic words, exactly one hit
 G4  primitivity
 G5  rotation fidelity: the drawn candidate == the base bitmap actually rotated
 G6  count entropy 0: the multiset of levels is identical in every figure
 G8  level legibility on delivered pixels: adjacent levels must differ by at
     least RAD_FLOOR px of bead travel and BLOB_FLOOR px^2 of connected change
 G11 the index badge does not touch a single coloured pixel of the dial
 word gates W1..W8                           see build/word_gates.py
"""
from __future__ import annotations
from . import dial as D

MIN_CC = 110              # G1
RAD_FLOOR = 8.0           # G8 adjacent-level bead travel, delivered px
BLOB_FLOOR = 55           # G8 adjacent-level symbol XOR, largest blob
MAX_ROTDIFF = 0.045       # G5


# ------------------------------------------------------------------- gates
def geometry_gates(L, col):
    """G8 and the fixed dial geometry, asserted for level count L in colour col."""
    g = D.geom()
    assert g["bead_pitch_min"] >= 2.5, g
    assert g["bar_pitch_hub"] >= 3.0, g
    assert g["rim_clear"] >= 1.5, g
    assert g["tile_margin"] >= 1.5, g
    assert g["hub_to_first"] >= 2.0, g
    leg = D.level_legibility(L, col)
    rad = min(x["d_rad"] for x in leg)
    blob = min(x["blob"] for x in leg)
    assert rad >= RAD_FLOOR, f"L={L} adjacent levels only {rad:.2f} px apart"
    assert blob >= BLOB_FLOOR, f"L={L} adjacent-level blob only {blob} px2"
