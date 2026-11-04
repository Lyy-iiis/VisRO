"""count_points_on_lines: the five tiers and the per-candidate gates.

TIERS is the only place difficulty is defined:

                      canvas   N   L   dot_r  near-miss band  on_tol  min near-miss
  easy               448x448  11   3   0.16   0.40-0.44       0.18      2
  medium             448x448  13   4   0.12   0.28-0.32       0.15      4
  hard               672x672  19  12   0.10   0.24-0.26       0.13      9
  xhard              672x672  20  11   0.10   0.24-0.26       0.13     10
  xxhard             672x672  20  13   0.10   0.24-0.26       0.13     10

N is the lattice side, L the number of segments; dot radius, near-miss band and
on-tolerance are fractions of the lattice spacing.  N starts at 11 because the
gates ban trivial slopes and require each segment to span at least three
primitive steps: the smallest non-trivial direction (1,2) needs N >= 7, and
N = 11 is the first lattice on which per-segment counts take three values.

gen_one(tier, seed) builds one candidate with cpl_build.make_problem and runs
the gates (one set for all tiers):

  G0  pixel budget    w%28==0 and h%28==0 and w*h<=451584
  G1  legibility      dot diameter >= 6 px, line width >= 2 px, OFF clearance
                      >= 3 px, the near-miss count, a full geometry recheck
  G2  tier shape      grid_n and n_segs equal the tier's values
  G3  ON legible      in the final PNG every ON dot has line ink on both sides
  G4  OFF white gap   the tightest OFF white gap >= 1.0 px
  G5  no false pos    no OFF dot satisfies the ON criterion
  G6  answer sound    red rings in the solution overlay == the answer total
  G7  endpoints       overshot endpoints are >= 0.80 * spacing apart
  G8  near-parallel   segments within 15 degrees are >= 1.0 * spacing apart
  G9  per-line unique three independent implementations agree
  G10 ON clearance    a segment that does not pass through an ON dot clears
                      it by >= 3 px
  G11 no trivial slope: no primitive direction (1,0), (0,1), (1,1), (1,-1)
  G12 non-degenerate  the per-segment counts are not all equal
"""
import random

from . import cpl_build, cpl_core, cpl_gates

# ==================================================================== TIERS table
# dial/segs are the same number (segment count L); N is the lattice side.
# size/dot_r/margin_lo/margin_hi/on_tol are the four acuity constants + canvas.
TIERS = {
    "easy":   dict(dial=3,  N=11, segs=3,  nearmiss=2,  size=448,
                   dot_r=0.16, margin_lo=0.40, margin_hi=0.44, on_tol=0.18),
    "medium": dict(dial=4,  N=13, segs=4,  nearmiss=4,  size=448,
                   dot_r=0.12, margin_lo=0.28, margin_hi=0.32, on_tol=0.15),
    "hard":   dict(dial=12, N=19, segs=12, nearmiss=9,  size=672,
                   dot_r=0.10, margin_lo=0.24, margin_hi=0.26, on_tol=0.13),
    "xhard":  dict(dial=11, N=20, segs=11, nearmiss=10, size=672,
                   dot_r=0.10, margin_lo=0.24, margin_hi=0.26, on_tol=0.13),
    "xxhard": dict(dial=13, N=20, segs=13, nearmiss=10, size=672,
                   dot_r=0.10, margin_lo=0.24, margin_hi=0.26, on_tol=0.13),
}

# ---- register the five tiers with cpl_build's DIAL table
#      (cpl_build.make_problem only reads dial/N/segs/nearmiss from it)
for _t, _c in TIERS.items():
    cpl_build.DIAL[_t] = dict(dial=_c["dial"], N=_c["N"], segs=_c["segs"],
                              nearmiss=_c["nearmiss"])


def size_gate(img):                      # G0
    w, h = img.size
    return (w % 28 == 0) and (h % 28 == 0) and (w * h <= 451584)


def g11_nontrivial_slopes(m):            # G11 no trivial slopes (recheck)
    for dx, dy in m["prim_dirs"]:
        k = (abs(dx), abs(dy))
        if k in {(1, 0), (0, 1), (1, 1)}:
            return False
    return True


def g12_nondegenerate(m):                # G12 per-line counts not all equal
    return len(set(m["per_line"])) >= 2


def gen_one(tier, seed):
    """Build one candidate problem and run G0..G12.  Returns (ok, why, payload)."""
    cfg = TIERS[tier]
    # ---- write this tier's four acuity constants back into cpl_build's module
    #      globals (make_problem reads them at call time)
    cpl_build.DOT_R_FRAC = cfg["dot_r"]
    cpl_build.ON_TOL = cfg["on_tol"]
    cpl_build.MARGIN_LO = cfg["margin_lo"]
    cpl_build.MARGIN_HI = cfg["margin_hi"]
    rng = random.Random(seed)
    try:
        r = cpl_build.make_problem(rng, tier, size=cfg["size"], force_color="blue")
    except cpl_core.Illegible:
        return False, "G1_illegible", None
    if r is None:
        return False, "G1_build", None
    img, m, segs, on_set, to_px, dot_r = r
    if m["grid_n"] != cfg["N"] or m["n_segs"] != cfg["segs"]:
        return False, "G2_shape", None
    if not size_gate(img):
        return False, "G0_size", None
    ok, why, diag = cpl_gates.pixel_gates(img, m, segs, on_set, to_px, dot_r)
    if not ok:
        return False, why, None
    g10ok, on_clear, g10at = cpl_gates.on_clearance_gate(m, segs, on_set, to_px, dot_r)
    diag["on_clear_px"] = on_clear
    diag["on_clear_at"] = g10at
    if not g10ok:
        return False, "G10_on_clear", None
    if not g11_nontrivial_slopes(m):
        return False, "G11_trivial_slope", None
    if not g12_nondegenerate(m):
        return False, "G12_degenerate", None
    return True, None, (img, m, segs, on_set, diag, to_px, dot_r)
