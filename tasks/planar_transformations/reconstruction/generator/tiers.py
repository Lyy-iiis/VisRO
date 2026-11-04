"""reconstruction -- the five-tier table, read by `decoys.build`.

    easy  <  medium  <  hard  <  xhard  <  xxhard

The four upper tiers are one "target rigidity" dial: they agree on every
parameter except the target rectangle(s) and rho (asserted at import time).

    tier     rho = cells/(W*H)   rects  cells   parts
    easy     0.5714               4      20-24    4
    medium   0.667                1         48    8
    hard     0.857                1         48    8
    xhard    0.980                8      47-54    8
    xxhard   1.000                8      48-55    8

TARGET SHAPES
-------------
  (a) `easy` uses 20-24 cells in 4 pieces on four rectangles; its rho 0.5714 is
      the loosest rung, below medium's 0.667.  It is a difficulty parameter,
      not a gate; every anti-shortcut gate listed below applies unchanged.

  (b) `xhard` and `xxhard` draw from a LIST of bounding rectangles.  Every
      entry of the list is put through the same rho line

          abs(cells / (w * h) - rho) < 0.002

      in `_freeze_check` below.  The list only widens WHICH SHAPES realise the
      tier's rho -- xxhard is still "a solid rectangle", just not always the
      SAME solid rectangle.  Distinct target shapes:

          xhard   210   (essential 47)
          xxhard    8   (essential  7)

      `medium` and `hard` have no shape list, and `decoys.propose` draws no
      rng for the shape when the key is absent.

      NOTE.  A solid rectangle is centrally symmetric, so ANY tiling of
      it turned 180 degrees is a second, equally valid way to place the same
      pieces.  Changing the aspect ratio cannot escape that -- it is a property
      of rectangles, not of one rectangle -- and for the square 7x7 there are
      four such placements rather than two.  This does not touch the answer:
      exactly ONE of the four options tiles the target, which is what the
      question asks, and that is re-verified per item by a fresh solver.

WHY THE LADDER IS MONOTONE BY CONSTRUCTION
------------------------------------------
Two independent knobs, both moving the same way:

  * READING LOAD.  easy shows 20-24 target cells cut into 4 pieces; every
    upper tier shows 47-55 cells cut into 8.  Strictly less picture to hold in
    mind -- asserted below at the EXTREMES of both shape lists, not just at
    their nominal rows.
  * TARGET RIGIDITY rho.  A bite taken out of the bounding rectangle creates
    concavity, and concavity is free information: a "spike" (a target cell
    with <= 1 neighbour inside the target) can be covered in only a handful of
    ways, and a deep notch can make a whole option die by pure look-up
    with no search at all.  rho climbs 0.5714 -> 1.000
    across the five tiers, so information is only ever REMOVED going up.

Refutation work (exhaustive node counts, the same yardstick at every tier)
separates the five tiers.  This file fixes their ORDER; the node counts depend
on the solver and are not restated here.

THE `easy` TIER
---------------
  * 20-24 cells in 4 pieces (min_size = 3).  A smaller target (e.g. 14 cells in
    3 pieces) has too few distinct images for a 25k-item cell; 20-24 cells
    lift that ceiling by orders of magnitude while leaving easy far cheaper to
    refute than medium.
  * four rectangles at one rho -- 7x6 and 6x7 (24 cells), 7x5 and 5x7
    (20 cells) -- with 15-18 boundary cells bitten off, by the SAME sampler
    (tiling.bite_rect) and the same criterion as the upper tiers, placed at
    the loose end of the same rho dial: rho = 24/42 = 20/35 = 0.5714 < medium's
    0.667.  This is what puts easy ON the ladder instead of beside it.
  * rotations = True: all five tiers ask the identical question, "The pieces
    may be moved and rotated but NOT flipped."
  * decoys are carved from the target like the upper tiers' (decoys.py) and
    carry the sig() of the piece they replace.  A decoy that changed a piece's
    cell count would leave the option sets with unequal total area, and "count
    the cells" would solve the item.
  * refute_band floor 40 -> 1.  This is the DIFFICULTY knob, not an
    anti-shortcut gate: it says "how much work must refuting a wrong option
    cost", and an easy tier is precisely one where that cost is small.  The
    floor is kept > 0, which still rejects the degenerate item where the very
    first cell of the target cannot be covered by any piece of the wrong
    option at all.

EVERY ANTI-SHORTCUT GATE IS KEPT VERBATIM FOR ALL FIVE TIERS, NONE RELAXED:
  gate 0  all 2n shapes canonically distinct
  gate R  target spans its full WxH bounding box and stays connected
  gate A  the truth is findable inside solve_band
  gate 1  every distractor refuted under ROTATIONS-ONLY, work inside refute_band
  gate 2  every distractor refuted again with MIRROR ALLOWED
  plus    decoys individually placeable inside the target, decoys carry the same
          sig() = (area, perimeter, sorted bbox) as the piece they replace, no
          duplicate option sets, pieces re-posed and shuffled inside every panel,
          answer position shuffled.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# THE TABLE.  Five rows, one generator.
# --------------------------------------------------------------------------- #
# The rectangles each tier may draw its target from.  EVERY entry realises the
# tier's own rho to the tier's own tolerance (asserted below with the identical
# expression), so this list widens WHICH SHAPES realise a rho and nothing else.
# `medium` and `hard` deliberately have NO entry: they stay on a single
# rectangle, and `decoys.propose` makes no rng draw for it when the key is absent.
#   * h <= 8 everywhere, so the target never renders below 147 // 8 = 18 px
#     per cell -- the smallest target cell size of any tier
#     (medium's 9x8).
#   * both dims >= 4, so a min_size piece always fits.
EASY_SHAPES = [(7, 6, 24), (6, 7, 24), (7, 5, 20), (5, 7, 20)]

# areas 46..55 are exactly the areas A for which (A-1)/A lands inside
# 0.980 +- 0.002; intersected with 4 <= h <= 8 and 4 <= w.
RIGID_RECTS = [(8, 6), (6, 8), (12, 4), (7, 7), (10, 5), (13, 4),
               (9, 6), (11, 5)]
XHARD_SHAPES = [(w, h, w * h - 1) for w, h in RIGID_RECTS]   # rho -> 0.980
XXHARD_SHAPES = [(w, h, w * h) for w, h in RIGID_RECTS]      # rho -> 1.000

TIERS = {
    "easy":   dict(rho=0.5714, rect=(7, 6), cells=24, n_parts=4, min_size=3,
                   max_size=10, rotations=True, decoy_attempts=8,
                   refute_band=(1, 50_000), solve_band=(1, 200_000),
                   mirror_cap=3_000_000, shapes=EASY_SHAPES),
    "medium": dict(rho=0.667, rect=(9, 8), cells=48, n_parts=8, min_size=4,
                   max_size=10, rotations=True, decoy_attempts=8,
                   refute_band=(40, 50_000), solve_band=(1, 200_000),
                   mirror_cap=3_000_000),
    "hard":   dict(rho=0.857, rect=(8, 7), cells=48, n_parts=8, min_size=4,
                   max_size=10, rotations=True, decoy_attempts=8,
                   refute_band=(40, 50_000), solve_band=(1, 200_000),
                   mirror_cap=3_000_000),
    "xhard":  dict(rho=0.980, rect=(7, 7), cells=48, n_parts=8, min_size=4,
                   max_size=10, rotations=True, decoy_attempts=8,
                   refute_band=(40, 50_000), solve_band=(1, 200_000),
                   mirror_cap=3_000_000, shapes=XHARD_SHAPES),
    "xxhard": dict(rho=1.000, rect=(8, 6), cells=48, n_parts=8, min_size=4,
                   max_size=10, rotations=True, decoy_attempts=8,
                   refute_band=(40, 50_000), solve_band=(1, 200_000),
                   mirror_cap=3_000_000, shapes=XXHARD_SHAPES),
}

ORDER = ["easy", "medium", "hard", "xhard", "xxhard"]


def cfg_for(tier):
    c = dict(TIERS[tier])
    c.pop("rho")
    return c


def shapes_for(tier):
    """Every (w, h, cells) the tier may draw, nominal row included."""
    c = TIERS[tier]
    return [tuple(s) for s in c.get("shapes",
                                    [(c["rect"][0], c["rect"][1], c["cells"])])]


def _freeze_check():
    """The upper tiers agree on every field except rect and rho (plus the
    added `shapes` list of xhard/xxhard, which cannot change rho, because
    every entry in it is put through the same rho criterion below)."""
    def dial(t):
        return {k: v for k, v in cfg_for(t).items() if k not in ("rect", "shapes")}
    for tier in ORDER[2:]:
        assert dial(tier) == dial("medium"), tier
    # monotone rho, monotone reading load
    rho = [TIERS[t]["rho"] for t in ORDER]
    assert rho == sorted(rho), rho
    cells = [TIERS[t]["cells"] for t in ORDER]
    assert cells == sorted(cells), cells
    parts = [TIERS[t]["n_parts"] for t in ORDER]
    assert parts == sorted(parts), parts
    for t in ORDER:
        w, h = TIERS[t]["rect"]
        assert abs(TIERS[t]["cells"] / (w * h) - TIERS[t]["rho"]) < 0.002, t
        # THE SAME CRITERION, THE SAME TOLERANCE, applied to every alternate
        # shape.  This is the whole of what the shape list widens: a shape is allowed
        # in a tier if and only if it passes the line above.
        for (aw, ah, ac) in shapes_for(t):
            assert abs(ac / (aw * ah) - TIERS[t]["rho"]) < 0.002, (t, aw, ah)
            assert 4 <= ah <= 8 and aw >= 4, (t, aw, ah)   # render + min_size
            assert ac >= TIERS[t]["n_parts"] * TIERS[t]["min_size"], (t, ac)
            assert ac <= TIERS[t]["n_parts"] * TIERS[t]["max_size"], (t, ac)
        assert (w, h, TIERS[t]["cells"]) in shapes_for(t), t
    # reading load stays strictly ordered even at the extremes of the shape
    # lists: the heaviest easy item is lighter than the lightest medium one.
    assert max(c for _, _, c in shapes_for("easy")) < \
        min(c for _, _, c in shapes_for("medium")), "easy reading load"


_freeze_check()
