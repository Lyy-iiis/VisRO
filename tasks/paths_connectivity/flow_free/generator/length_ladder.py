"""flow_free length ladder: the five four_pipe tiers, separated by PIPE LENGTH.

WHY THE TIERS BAND ON LENGTH.
  * Pipe LENGTH is the real difficulty axis; a turn count is largely a proxy
    for it.  So the tiers band on length whichever statistic the question asks
    for.
  * On the 448 canvas, changing n changes PIXELS, and that alone moves accuracy
    a lot.  So the canvas is fixed at 448x448 and the three deep tiers share ONE
    n (15), one pipe count (4) and therefore one pixel geometry (26.9 px cell /
    10 px stroke); only the length band moves.  easy and medium keep mixed sizes
    (9+11 and 11+13).

THE FIVE BANDS.  Acceptance rates measured at calibration, for the chain >=3
segments, in band, a non-adjacent same-colour distractor of >=10 cells, a
dE>=55 palette, a ring that reads on the target colour, and a pixel positive
control at the ring:

    easy   12-22   n=9  0.22   n=11 0.29
    medium 25-38   n=11 0.25   n=13 0.28
    hard   40-55   n=15 0.17
    xhard  70-90   n=15 0.04
    xxhard 110-135 n=15 0.04

Two bands are physically empty: n=9 cannot make a 40-55 cell pipe at ANY pipe
count (the target would eat 49-68% of an 81-cell board and the distractor then
has nowhere to go), and n=13 cannot make a 110-135 cell pipe (only c=3 reaches
the length, and at c=3 the three segments are adjacent by construction).  That is
why the deep tiers are n=15.

What the tier reveals: the pipe count is pinned to 4 on every tier, the grid
size differs only between easy/medium and the deep tiers, and the length band
is the tier by construction -- knowing the tier does not pin the answer,
because the five answer ranges overlap pairwise.
"""
from __future__ import annotations


# --------------------------------------------------------------- the recipe
# Every constant below defines the ladder; they are literals here, not knobs.

TORT = 0.95                 # ALL five tiers, including n=9/11/13
PIPES = 4                   # pinned across the ladder -> I(num_pipes; tier) = 0

LEN_BANDS = {
    "easy":   (12, 22),
    "medium": (25, 38),
    "hard":   (40, 55),
    "xhard":  (70, 90),
    "xxhard": (110, 135),
}
TIER_NS = {
    "easy":   (9, 11),
    "medium": (11, 13),
    "hard":   (15,),
    "xhard":  (15,),
    "xxhard": (15,),
}
TIERS = ("easy", "medium", "hard", "xhard", "xxhard")
