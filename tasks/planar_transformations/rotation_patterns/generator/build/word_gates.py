"""The word gates W1..W8 (constants CFG and MAX_PRED) shared by the pool builders.

Combinatorial gates
  W1 primitive                the 16 shifts are pairwise different
  W2 every level used >= 2x   a level used once would BE the landmark
  W3 m_anchor >= m_req        shortest unique CONTIGUOUS window.  m_req is 3 at
     L2 and 2 above it, because that is the alphabet's own floor: with L levels
     a spoke carries log2(L) bits, and at L>=3 two spokes already carry nearly
     the log2(16)=4 bits that pin the phase, so 99% of L3 words and 100% of L5
     words have m_anchor exactly 2.  W7 is what keeps the reading honest: a
     solver who can only tell long from short still has to read 3 in a row.
  W4 longest run <= run_cap, and the longest run occurs at least twice
  W5 min pairwise Hamming over the 16 shifts >= min_ham
  W6 gestalt: the argmax/argmin of (spoke), (adjacent pair), (adjacent step),
     (triple) are each attained at least twice
  W7 coarse: EVERY threshold binarisation still needs >= 3 consecutive spokes

The centroid gate:
  W8 the ideal ink-centroid offset of the figure <= MAX_PRED delivered px.
     A candidate is the base rotated, so its centroid bearing rotates with it and
     "read the bearing, subtract the base's, divide by 22.5" hands over the shift
     with no reading at all.  Exact cancellation is impossible for this symbol
     (proof in moment.py), so the gate pushes |A_1| down to its floor instead,
     where the offset (~0.1 px) is buried in rasterisation noise and the probe
     falls back to chance.

W8 is by far the rarest gate AND the cheapest to evaluate (16 complex
multiplies), so the search tests it first; only ~1% of sampled words ever pay for
the O(k^2) combinatorial stack.
"""
from __future__ import annotations

MAX_PRED = 0.20          # delivered px: W8's bound on the ideal ink-centroid offset

CFG = {
    2: dict(m_req=3, run_cap=4, min_ham=4, trials=0,      seed=0),
    3: dict(m_req=2, run_cap=3, min_ham=5, trials=4000000, seed=3),
    4: dict(m_req=2, run_cap=3, min_ham=6, trials=2000000, seed=4),
    5: dict(m_req=2, run_cap=3, min_ham=7, trials=3000000, seed=5),
}
