# -*- coding: utf-8 -*-
"""overlap_match -- five-tier ladder on signature-arc burial depth B*.

TIER MAP.  One TIERS table, five rows:

    D0 -> easy     D1 -> medium   D2 -> hard   D3 -> xhard   D4 -> xxhard

WHAT THE TASK IS.  896x504, white ground.  The left 392x504 is the STACK: n
outline polygons drawn on top of one another.  The right half holds four
candidate cells A/B/C/D.  Exactly one candidate occurs inside the stack,
translated only -- never rotated, never rescaled.  The answer is one letter.

The four candidates share one base outline and differ only at 4 SIGNATURE LOCI
(notches cut into edges).  The loci are split into 2 blocks of 2, the two loci
in a block being redundant copies of each other, so a candidate is the word
(a,a,b,b).  Two of the four signature arcs can therefore be buried to solid
black and the answer is still unique, as long as one copy per block stays
readable.  That redundancy is what makes burying the evidence safe.

THE AXIS.  B* = 1 - mean(visible fraction of the target
host's 4 signature arcs), measured on the delivered geometry.  Each tier is a
band on B*: measured 0.13 / 0.35 / 0.56 / 0.69 / 0.82.

Two other scales are only DECIDABILITY FLOORS, never the dial: r* (the largest
read tolerance under which the answer is still unique) and E*(2px) (the arc the
most-deniable wrong candidate must still expose).  r* sits at the pixel floor
already on easy boards, so a dial on it would make tiers harder by deleting the
answer rather than by demanding more reasoning.  Burial depth is what varies
between boards of one scale, so it is the controlled knob.

COUPLED MONOTONE PARAMETERS (all move with the tier; none is a shared const):
    n     shapes in the stack     6 / 8 / 9 / 10 / 11
    H     same-base decoy hosts   1 / 1 / 2 /  3 /  4
    dpx   candidate separability band, px
    maxvis  cap on ANY shape's visible fraction -- kills "compare against the
            one clean silhouette lying around".
    rec   band on the target's overall visible fraction
    estar band on E*(2px)

STRUCTURE -- all of it add-only gating:
1.  4 loci in 2 redundant blocks (above).  Reading a single locus can only cut
    4 candidates down to 2+2.
2.  The H same-base DECOY HOSTS take the block-inconsistent words
    (a,1-a,b,1-b): Hamming distance exactly 2 from every candidate, so a coarse
    look at the whole silhouette treats all H+1 hosts alike and the loci have
    to be read one at a time.
3.  NOISO / CROSS gates: every shape is crossed by at least cross_min others.
    maxvis gate: no clean whole silhouette anywhere in the picture.
4.  DECIDABILITY FLOORS, identical in all five tiers, never relaxed.  Under an
    exhaustive integer-translation scan (FFT, every offset, including offsets
    that hang off the canvas):
      - the correct candidate is fully explained by stack ink at the host;
      - every wrong candidate, at its OWN most favourable offset anywhere in
        translation space, still exposes >= E_FLOOR px of arc, containing a
        >= RUN_FLOOR px contiguous run, at least one point of which is
        >= DEPTH_FLOOR px away from ANY ink.
    DEPTH_FLOOR = 5px is wider than the 3px stroke, so that white gap is a gap
    a person can see, not a sub-pixel judgement.  This is the machine guarantee
    that a low score means difficulty and not an unreadable image.
5.  Also: DPX/DCAP/DRATIO separability band, decoy planting
    (every signature value also occurs on filler shapes), OCCRANK (the target's
    visibility may not be the extreme of the board), PILEVIS (the stack must be
    connected), answer-letter quota, fixed seed + md5 re-run.

Canvas 896x504 = 451,584 px: exactly the 576-visual-token budget of this arm,
and both sides divide by 28.

PLACEMENT LAW.  Target and non-targets draw their positions from the same
free_pos(), with the annulus scale dref = mean diameter over ALL shapes, and the
target sticks to a decoy host's signature arc with the same probability and the
same geometry as any other shape (it is in the hill-climb's movable set).  So no
probe that reads positions only -- "the shape closest to the centre of the
stack" -- beats chance.  Gate G_pos checks a per-item structural invariant plus
a batch statistical test over the probe family P_center / P_far / P_centroid /
P_medoid / P_top / P_bottom / P_left / P_right / P_*_bb / P_zfirst / P_zlast,
each of which must sit at 1/n.  P_big / P_small are reported, not gated,
because they are set by geometric scale rather than by placement.

SIZE LAW.  build_item() draws every host's and every filler's bounding box from
the four CANDIDATE words' boxes and puts the shape on it exactly, so all n boxes
on a board are i.i.d. from one distribution and size says nothing about which
shape is the target.

CANDIDATE PANELS AND UNIQUENESS MARGIN.
  * panel_offset() picks one registration per item so all four outlines
    (stroke + PANEL_CLEAR) fit their frame and clear the A-D letter;
    build_item() refuses a candidate set that cannot fit;
  * make() requires gap_gate() (every wrong candidate keeps an outline point
    >= GAP_FLOOR px from the stack ink at EVERY translation on a 0.5px lattice)
    and r* >= RSTAR_FLOOR.

D0 (easy) follows the base generator's easy settings (two shapes, occlusion
fraction 0.013..0.170, candidate Hausdorff separation 0.10..0.27 diameters),
with n=6 rather than 2 and occlusion expressed in this file's stricter rec
measure (argued at the D0 row of TIERS below).
"""
from __future__ import annotations
from visro.seeding import seed_text

import math
from itertools import product

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from .base import polygons


# ----------------------------------------------------------- layout / pixels
OUT_W, OUT_H = 896, 504
CELL = OUT_H // 2                 # 252
STACK_W = OUT_W - 2 * CELL        # 392
PAD = 8
SS = 2
STROKE_OUT = 3
BASE_R = 129.0
MAX_DIM = 208
# SIZE LAW (build_item).  A non-target shape is put on
# the EXACT bounding box of one of the four candidate words.  Fillers are
# generated fresh, so they are resampled until an isotropic fit leaves at most
# FILL_ANISO of shape left over; the H same-base decoy hosts cannot be
# resampled, so ANISO_CAP bounds what they may be stretched by and the item is
# thrown away when a host needs more.  Both are logs of a ratio of scale
# factors: 0.02 of a 208px outline is ~2px of differential stretch at the
# extremes, i.e. under the 3px stroke.
FILL_ANISO = 0.02
ANISO_CAP = 0.05

# CANDIDATE PANEL LAYOUT.  Combos reach MAX_DIM*1.12 = 233px and a spike-out is
# one-sided, so a fixed registration would let outlines overflow the 234px frame
# or cross the A-D letter.  The four candidates share ONE registration (the base
# outline sits at the same place in all four cells), chosen per item by
# panel_offset() so that the union of the four outlines, stroke included,
# lies inside the frame with PANEL_CLEAR px of white to spare and clears the
# letter glyph by the same margin.  A candidate set that cannot fit is refused
# in build_item() (STATS "panel_fit"), before any placement work is spent.
PANEL_BOX = CELL - 2 * PAD - 2          # 234, side of the rounded frame
FRAME_W = 2                             # frame line width, drawn inward
PANEL_CLEAR = 2.0                       # px of white: stroke vs frame / letter
LETTER_BOX = (8.0, 10.0, 22.0, 25.0)    # area of the A-D glyph's ink in frame
                                        # coords: pixels 8..21 x 10..24 (Pillow
                                        # 12.3 default font at 20px, which
                                        # render() puts at (8, 5))

# UNIQUENESS MARGIN.  solve() proves every wrong candidate exposes >= DEPTH_FLOOR
# px of gap, but only at its r=2 count-minimising offsets; at some OTHER offset a
# wrong candidate could sit with every outline point within 3px of ink.  So
# make() also requires that, at EVERY translation on a 0.5px lattice inside the
# stack panel, every wrong candidate has an outline point >= GAP_FLOOR px from
# the nearest stack ink pixel (gap_gate() / gap_margins(), on the delivered
# PNG), and that r* >= RSTAR_FLOOR (rstar_of()).
# GAP_FLOOR = 3.5 admits EDT values 3.61, 4.0, ...: with ink pixels reaching
# ~1.2px from a stroke's centreline, the wrong outline's centreline passes
# ~4.8px from every drawn line at that point, i.e. ~1.8px of white between the
# two 3px strokes.  A floor of 4.0 would reject most xxhard boards.
GAP_FLOOR = 3.5
RSTAR_FLOOR = 3

# ------------------------------------ evidence-quality floors, never relaxed
R_OCC = 5.0            # within 5px of another stroke's centreline: unreadable
R_READ = 2             # read tolerance (3px stroke + LANCZOS antialiasing)
E_FLOOR = 3.0          # arc a wrong candidate must expose at its best offset(px)
RUN_FLOOR = 4          # of which one contiguous white run (px)
DEPTH_FLOOR = 5.0      # that run needs a point >= 5px from ANY ink pixel.
                       # Ink already includes the 3px stroke, so this is 5px
                       # clear of the stroke's outer edge; drawing the candidate
                       # there still leaves ~3.5px of real white -- a gap the eye
                       # sees, not a sub-pixel call.  This is the hard guarantee
                       # that 0% is never "too blurred to read".
SEP_MIN = 6.0          # per-point separation floor between a locus' two values
SEP_FRAC = 0.45        # at least this fraction of arc points must reach SEP_MIN
                       # The key gate of this design: once a locus is mostly
                       # buried, whatever short piece of arc still shows must
                       # stay DEEPLY discriminative, or burial would degrade
                       # falsification into a 2-3px sub-pixel judgement.
DCAP = 40.0            # separability cap: no candidate may differ at a glance
DRATIO = 2.2           # max/min pairwise gap: no candidate is the odd one out
APEX_R = 7
SHOULDER = 31
MIN_LOCUS_EDGE = 56.0
MIN_DECOY_EDGE = 68.0
MIN_BASE_EDGE = 23.0
MIN_NOTCH_W = 8.0
NOTCH_DEPTH = (13.0, 26.0)
NOTCH_W = (0.150, 0.200)
L_LOCI = 4             # locus count: a structural constant, same in every
                       # tier.  Answer length is NOT the axis.
K_VAL = 2
# BLOCK CODE.  The 4 loci are split into 2 blocks of 2; the two loci in a block
# are redundant copies of each other.
#   candidate   = (a,a,b,b), a,b in {0,1}  -> 4 words, pairwise Hamming 2 or 4
#   decoy host  = the 4 words that contradict themselves inside BOTH blocks,
#                 (a,1-a,b,1-b) -> Hamming exactly 2 from every candidate, so a
#                 host is exactly as hard to rule out as a candidate is
# This is what makes deep burial safe: as long as ONE copy per block is still
# readable the answer stays unique, so 2 of the 4 signature arcs may be buried
# to solid black and the exhaustive solver can still prove uniqueness -- and a
# person still has a chain of evidence they can walk to the end.

# ------------------------------------------------------------------ the dial
# vstar = min over blocks of (visible fraction of the block's most readable
#         signature arc) -- "how much of the best surviving evidence is left".
#         Decidability rests on it, so it is the real hard constraint.
# avis  = mean visible fraction over the 4 signature arcs;
#         B* = 1 - avis = burial depth, the reading of the main axis.
TIERS = {
    # D0 = easy.  Its parameters follow the base generator's easy settings,
    #   with two adaptations:
    #   (a) n = 6 rather than 2.  The anti-shortcut decoy-planting gate needs
    #       all L_LOCI*K_VAL = 8 signature values planted on filler shapes, at
    #       most 2 per filler, so n_fill >= 4; build_item() returns None when
    #       n_fill < 4, and with H = 1 the smallest workable n is 1+1+4 = 6.
    #       n = 2 produces no item at all.
    #   (b) Occlusion is expressed in this file's rec measure (fraction of
    #       outline points >= R_OCC = 5px from another stroke's centreline),
    #       which is far stricter than the base generator's occlusion_frac
    #       (stroke pixels literally painted over).  rec = (0.72, 0.92) converts
    #       to 0.038..0.095 (mean 0.067) on that measure, inside the base easy
    #       range 0.013..0.170.
    #   Every other entry is on the LOOSER side of D1, in the same direction as
    #   the coupled monotone parameters: n 6 < 8, H equal, ov wider, dpx larger
    #   (candidates differ more), vstar/avis higher (loci barely buried), rec
    #   higher (target barely occluded), maxvis higher, estar lower bound higher
    #   (wrong candidates expose more, so they are easier to falsify),
    #   chain_run longer, cross_min smaller, pfocus smaller.
    "D0": dict(n=6,  H=1, ov=(0.14, 0.42),  dpx=(16.0, 26.0),
               vstar=(0.90, 1.00), avis=(0.84, 1.00),
               rec=(0.72, 0.92), maxvis=0.99, estar=(22.0, 240.0),
               chain_run=20, cross_min=1, pfocus=0.10, fsim=0.00),
    "D1": dict(n=8,  H=1, ov=(0.05, 0.17),  dpx=(12.5, 20.0),
               vstar=(0.72, 0.96), avis=(0.58, 0.80),
               rec=(0.40, 0.60), maxvis=0.68, estar=(14.0, 90.0),
               chain_run=18, cross_min=2, pfocus=0.15, fsim=0.00),
    "D2": dict(n=9,  H=2, ov=(0.045, 0.16), dpx=(11.5, 17.5),
               vstar=(0.50, 0.68), avis=(0.40, 0.55),
               rec=(0.33, 0.50), maxvis=0.60, estar=(10.0, 70.0),
               chain_run=14, cross_min=2, pfocus=0.35, fsim=0.34),
    "D3": dict(n=10, H=3, ov=(0.04, 0.15),  dpx=(10.5, 16.0),
               vstar=(0.34, 0.47), avis=(0.24, 0.36),
               rec=(0.27, 0.42), maxvis=0.52, estar=(7.0, 55.0),
               chain_run=10, cross_min=3, pfocus=0.50, fsim=0.67),
    "D4": dict(n=11, H=4, ov=(0.035, 0.14), dpx=(9.5, 15.0),
               vstar=(0.20, 0.32), avis=(0.10, 0.21),
               rec=(0.21, 0.34), maxvis=0.45, estar=(4.0, 45.0),
               chain_run=7,  cross_min=3, pfocus=0.62, fsim=1.00),
}

QUESTION = (
    'Exactly one of the shapes A-D on the right appears in the overlapping outlines on the left, '
    'at the same size and orientation (moved only, never rotated or resized). Which one? Answer '
    'with the letter.'
)
# The stack is the left 392 px and the candidates are the 2x2 cells on the right.

STATS = {}
# fixed RNG namespace of every board (changing it changes every item)
_DIAL_STREAM = seed_text(431264721970)


def bump(k, n=1):
    STATS[k] = STATS.get(k, 0) + n


# ------------------------------------------------------- geometry helpers
def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def seg_hit(p1, p2, p3, p4):
    d1 = _cross(p4[0] - p3[0], p4[1] - p3[1], p1[0] - p3[0], p1[1] - p3[1])
    d2 = _cross(p4[0] - p3[0], p4[1] - p3[1], p2[0] - p3[0], p2[1] - p3[1])
    d3 = _cross(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1])
    d4 = _cross(p2[0] - p1[0], p2[1] - p1[1], p4[0] - p1[0], p4[1] - p1[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple(pts):
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if seg_hit(a, b, c, d):
                return False
    return True


def point_in_poly(pt, pts):
    x, y = pt
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def bbox_wh(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def center_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [(x - cx, y - cy) for x, y in pts]


def fit(pts, dim=MAX_DIM):
    p = center_bbox(pts)
    m = max(bbox_wh(p))
    return center_bbox([(x * dim / max(m, 1e-6), y * dim / max(m, 1e-6))
                        for x, y in p])


def fit_box(pts, W, H):
    """Scale a centred outline so its bounding box is EXACTLY W x H.

    fit() forces max(bbox) and leaves the other side to the shape; this forces
    both, which is what makes a non-target shape's (w, h, area, aspect) land on
    a candidate word's values with nothing left over.  The correction away from
    isotropic is bounded by the caller (FILL_ANISO / ANISO_CAP).
    """
    p = center_bbox(pts)
    w, h = bbox_wh(p)
    sx, sy = W / max(w, 1e-6), H / max(h, 1e-6)
    return center_bbox([(x * sx, y * sy) for x, y in p])


def rot90(pts):
    """Quarter turn.  A rotation, not a reflection, so a signature value
    planted on a filler stays a congruent copy of that value.  Used only to put
    a filler's long axis on the side the drawn bounding box wants; fillers are
    already sampled at arbitrary orientations."""
    return [(-y, x) for x, y in pts]


def aniso(a, b):
    """|log| of the ratio of two aspect ratios: how far from isotropic the
    correction from aspect a to aspect b is."""
    return abs(math.log(max(a, 1e-9) / max(b, 1e-9)))


def trim_toward(a, b, keep):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    f = max(0.0, 1.0 - keep / max(L, 1e-6))
    return ((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), b)


def arr(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def seg_samples(a, b, step=1.0):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    k = max(2, int(L / step))
    return [(a[0] + (b[0] - a[0]) * t / k, a[1] + (b[1] - a[1]) * t / k)
            for t in range(k + 1)]


def outline_samples(pts, step=1.0):
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(2, int(L / step))
        for t in range(k):
            out.append((a[0] + (b[0] - a[0]) * t / k,
                        a[1] + (b[1] - a[1]) * t / k))
    return arr(out)


def panel_offset(polys):
    """Shared registration of the four candidate outlines inside their frame.

    Returns (ox, oy): where the outlines' common origin goes, in frame
    coordinates, on a 0.5px grid -- or None when the union of the four
    outlines (stroke + PANEL_CLEAR) cannot be put inside the frame while
    clearing the letter glyph by the same margin.  Preference: the union
    centred in the frame; otherwise the nearest 0.5px offset that clears the
    letter (ties broken by y, then x, so the choice is deterministic).  The
    union of the candidate SET is used, never the answer's own outline, so the
    registration carries no information about which candidate is correct."""
    half = STROKE_OUT / 2.0
    S = np.concatenate([outline_samples(p, 0.5) for p in polys])
    lo = FRAME_W + half + PANEL_CLEAR
    hi = PANEL_BOX - FRAME_W - half - PANEL_CLEAR
    (xmn, ymn), (xmx, ymx) = S.min(0), S.max(0)
    xs = np.arange(math.ceil((lo - xmn) * 2), math.floor((hi - xmx) * 2) + 1) / 2
    ys = np.arange(math.ceil((lo - ymn) * 2), math.floor((hi - ymx) * 2) + 1) / 2
    if not len(xs) or not len(ys):
        return None
    cx = PANEL_BOX / 2.0 - (xmn + xmx) / 2.0
    cy = PANEL_BOX / 2.0 - (ymn + ymx) / 2.0
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    order = np.lexsort((X, Y, np.round(np.hypot(X - cx, Y - cy), 6)))
    bx0, by0, bx1, by1 = LETTER_BOX
    keep = half + PANEL_CLEAR
    # only samples that can reach the letter at some feasible offset matter
    near = S[(S[:, 0] + xs[0] <= bx1 + keep) & (S[:, 1] + ys[0] <= by1 + keep)]
    for j in order:
        ox, oy = float(X[j]), float(Y[j])
        if len(near):
            px, py = near[:, 0] + ox, near[:, 1] + oy
            dx = np.maximum(np.maximum(bx0 - px, px - bx1), 0.0)
            dy = np.maximum(np.maximum(by0 - py, py - by1), 0.0)
            if float(np.hypot(dx, dy).min()) < keep:
                continue
        return (ox, oy)
    return None


def disc_samples(c, r, step=2.0):
    out = []
    n = max(3, int(r / step))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = i * step, j * step
            if x * x + y * y <= r * r:
                out.append((c[0] + x, c[1] + y))
    return out


def segs_of(pts, off=(0.0, 0.0)):
    P = np.asarray(pts, dtype=np.float64)
    A = P + np.asarray(off, dtype=np.float64)
    return A, np.roll(A, -1, axis=0)


def dist_to_segs(P, A, B):
    if len(A) == 0:
        return np.full(len(P), 1e9)
    d = B - A
    dd = (d * d).sum(-1)
    dd[dd == 0] = 1e-9
    w = P[:, None, :] - A[None, :, :]
    t = np.clip((w * d[None, :, :]).sum(-1) / dd[None, :], 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)


def runs_of(v):
    if v.size == 0:
        return 1.0, 0
    pad = np.concatenate(([False], v, [False]))
    dd = np.diff(pad.astype(np.int8))
    st = np.flatnonzero(dd == 1)
    en = np.flatnonzero(dd == -1)
    Lr = en - st
    return float(v.mean()), (int(Lr.max()) if Lr.size else 0)


def circ_max_run(v):
    if v.size == 0 or not v.any():
        return 0
    if v.all():
        return int(v.size)
    k = int(np.argmin(v))
    w = np.roll(v, -k)
    return runs_of(w)[1]


def hd_px(a, b):
    return float(polygons.hausdorff(polygons.poly_stroke_points(a, step=1.0),
                                    polygons.poly_stroke_points(b, step=1.0)))


def hd_fast(a, b):
    """Coarse Hausdorff (step=3px), for one-sided "far enough apart?" screens."""
    return float(polygons.hausdorff(polygons.poly_stroke_points(a, step=3.0),
                                    polygons.poly_stroke_points(b, step=3.0)))


def jitter_base(rng, base, frac):
    """Local vertex jitter (no Hausdorff
    retry loop); produces filler shapes whose silhouette reads like the base
    but whose detail differs."""
    d = polygons.diameter(base)
    for _ in range(30):
        out = list(base)
        for i in rng.sample(range(len(base)), rng.randint(2, 3)):
            ang = rng.uniform(0, 2 * math.pi)
            out[i] = (out[i][0] + frac * d * math.cos(ang),
                      out[i][1] + frac * d * math.sin(ang))
        if is_simple(out):
            return center_bbox(out)
    return None


# ------------------------------------------------------ signature features
class Notch:
    def __init__(self, t, w, depth, sign):
        self.t, self.w, self.depth, self.sign = t, w, depth, sign

    def clone(self):
        return Notch(self.t, self.w, self.depth, self.sign)

    def name(self):
        return (f"{'spike-out' if self.sign > 0 else 'notch-in'}"
                f" d={self.depth:.0f}px t={self.t:.2f} w={self.w:.3f}")


def apply_notch(pts, ei, nt):
    n = len(pts)
    P, Q = pts[ei], pts[(ei + 1) % n]
    L = math.hypot(Q[0] - P[0], Q[1] - P[1])
    w = nt.w * L
    if L < 4.0 * w or w < MIN_NOTCH_W:
        return None
    ux, uy = (Q[0] - P[0]) / L, (Q[1] - P[1]) / L
    nx, ny = -uy, ux
    mx, my = P[0] + nt.t * L * ux, P[1] + nt.t * L * uy
    if point_in_poly((mx + nx * 2.0, my + ny * 2.0), pts):
        nx, ny = -nx, -ny
    s = nt.sign * nt.depth
    A = (P[0] + (nt.t * L - w) * ux, P[1] + (nt.t * L - w) * uy)
    B = (P[0] + (nt.t * L + w) * ux, P[1] + (nt.t * L + w) * uy)
    apex = (mx + nx * s, my + ny * s)
    out = list(pts[:ei + 1]) + [A, apex, B] + list(pts[ei + 1:])
    return out if is_simple(out) else None


def apply_many(base, items):
    """Insert notches in descending edge order and shift already-recorded
    marks by +3, so the probes of every locus point at its own vertices."""
    out = list(base)
    marks = {}
    for ei, nt in sorted(items, key=lambda z: -z[0]):
        out2 = apply_notch(out, ei, nt)
        if out2 is None:
            return None, None
        out = out2
        for k in list(marks):
            marks[k] = tuple(i + 3 for i in marks[k])
        marks[ei] = (ei + 1, ei + 2, ei + 3)
    return out, marks


def sample_notch(rng):
    return Notch(rng.uniform(0.34, 0.66), rng.uniform(*NOTCH_W),
                 rng.uniform(*NOTCH_DEPTH), rng.choice((-1, 1)))


def mutate(rng, v0, m, sep, L):
    """Build a variant at a PRESCRIBED target separation sep (px).

    shift/widen are lateral moves along the edge, so separation is ~sep at
    almost every arc point (they pass the SEP gate easily); deep rotates the
    arm, so separation grows from 0 at the root to sep at the tip; flip is only
    available when sep is large.  sep is pinned into the tier's dpx band, so
    "how many pixels apart two candidates are" is designed, not sampled."""
    t, w, d, sgn = v0.t, v0.w, v0.depth, v0.sign
    if m == "shift":
        dt = (sep / L) * rng.choice((-1, 1))
        tt = t + dt
        if not (0.18 <= tt <= 0.82):
            tt = t - dt
        if not (0.18 <= tt <= 0.82):
            return None
        return Notch(tt, w, d, sgn)
    if m == "deep":
        for sg in (1, -1):
            nd = d + sg * sep
            if 11.0 <= nd <= 44.0:
                return Notch(t, w, nd, sgn)
        return None
    if m == "flip":
        nd = sep - d
        if not (11.0 <= nd <= 32.0):
            return None
        return Notch(t, w, nd, -sgn)
    nw = w + sep / L
    if nw > 0.235:
        nw = w - sep / L
    if nw < 0.10:
        return None
    return Notch(t, nw, d, sgn)


def arc_pts_of(poly, e):
    """Sample points of the signature arc (both arms) on edge e of poly."""
    ia, iap, ib = e + 1, e + 2, e + 3
    return arr(seg_samples(poly[ia], poly[iap], 1.0) +
               seg_samples(poly[iap], poly[ib], 1.0))


def sep_ok(p, q, e):
    """SEP gate: at least SEP_FRAC of p's signature-arc points are >= SEP_MIN
    from the whole of q's outline.  Both directions must pass, so whichever
    short piece of arc happens to stay visible is still DEEP evidence."""
    A, B = segs_of(q)
    f1 = float((dist_to_segs(arc_pts_of(p, e), A, B) >= SEP_MIN).mean())
    A, B = segs_of(p)
    f2 = float((dist_to_segs(arc_pts_of(q, e), A, B) >= SEP_MIN).mean())
    return min(f1, f2) >= SEP_FRAC


def variant_set(rng, base, e, dlo, dhi, K=K_VAL):
    """Sample 2 signature values on edge e whose separation lands in
    [dlo, dhi] and which pass the SEP gate."""
    nb = len(base)
    L = math.hypot(base[(e + 1) % nb][0] - base[e][0],
                   base[(e + 1) % nb][1] - base[e][1])
    for _a in range(120):
        v0 = sample_notch(rng)
        p0 = apply_notch(base, e, v0)
        if p0 is None:
            continue
        for _t in range(24):
            m = rng.choice(("shift", "shift", "widen", "deep", "flip"))
            sep = rng.uniform(dlo * 1.10, dhi * 0.90)
            v1 = mutate(rng, v0, m, sep, L)
            if v1 is None:
                continue
            p1 = apply_notch(base, e, v1)
            if p1 is None:
                continue
            if sep_ok(p1, p0, e) and dlo <= hd_fast(p1, p0) <= dhi:
                return [v0, v1]
    return None


def build_family(rng, T):
    """base + 4 loci x 2 values + all 16 combinations."""
    dlo, dhi = T["dpx"]
    for _ in range(500):
        bump("fam_try")
        base = fit(polygons.sample_polygon(rng, n_vert=rng.randint(8, 9),
                                           radius=BASE_R))
        nb = len(base)
        lens = [math.hypot(base[(i + 1) % nb][0] - base[i][0],
                           base[(i + 1) % nb][1] - base[i][1])
                for i in range(nb)]
        ok = [i for i in range(nb) if lens[i] >= MIN_LOCUS_EDGE]
        if len(ok) < L_LOCI:
            continue
        rng.shuffle(ok)
        edges = sorted(ok[:L_LOCI])
        variants, good = [], True
        for e in edges:
            vs = variant_set(rng, base, e, dlo, dhi)
            if vs is None:
                good = False
                break
            variants.append(vs)
        if not good:
            continue
        combos, marks = {}, {}
        bad = False
        for key in product(range(K_VAL), repeat=L_LOCI):
            sel = [(edges[l], variants[l][key[l]]) for l in range(L_LOCI)]
            pp, mk = apply_many(base, sel)
            if pp is None or max(bbox_wh(pp)) > MAX_DIM * 1.12:
                bad = True
                break
            combos[key] = pp
            marks[key] = mk
        if bad:
            continue
        # The 4 signature tips must stay apart, or two loci merge visually
        ap = [combos[(0,) * L_LOCI][marks[(0,) * L_LOCI][e][1]] for e in edges]
        if min(math.hypot(ap[i][0] - ap[j][0], ap[i][1] - ap[j][1])
               for i in range(L_LOCI) for j in range(i + 1, L_LOCI)) < 34.0:
            continue
        xs = [q[0] for q in combos[(0,) * L_LOCI]]
        ys = [q[1] for q in combos[(0,) * L_LOCI]]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        for k in combos:
            combos[k] = [(x - cx, y - cy) for x, y in combos[k]]
        base = [(x - cx, y - cy) for x, y in base]
        bump("fam_pass")
        return dict(base=base, edges=edges, variants=variants,
                    combos=combos, marks=marks)
    return None


def pick_sets(rng, fam, T):
    """Block code: split the 4 loci at random into 2 blocks of 2, the two loci
    of a block being redundant copies.  The candidates are the 4 words that
    agree inside both blocks; the decoy hosts are the 4 words that contradict
    themselves inside BOTH blocks -- Hamming distance exactly 2 from every
    candidate, so a host is as hard to tell from a candidate as candidates are
    from each other."""
    perm = list(range(L_LOCI))
    rng.shuffle(perm)
    blocks = [(perm[0], perm[1]), (perm[2], perm[3])]

    def word(a, b, flip):
        w = [0] * L_LOCI
        w[blocks[0][0]] = a
        w[blocks[0][1]] = a if not flip[0] else 1 - a
        w[blocks[1][0]] = b
        w[blocks[1][1]] = b if not flip[1] else 1 - b
        return tuple(w)

    cands = [word(a, b, (0, 0)) for a in (0, 1) for b in (0, 1)]
    hosts_all = [word(a, b, (1, 1)) for a in (0, 1) for b in (0, 1)]
    polys = [fam["combos"][c] for c in cands]
    gaps = [hd_px(polys[i], polys[j])
            for i in range(4) for j in range(i + 1, 4)]
    dlo, dhi = T["dpx"]
    if min(gaps) < dlo * 0.92 or max(gaps) > DCAP:
        return None
    if max(gaps) > DRATIO * min(gaps):
        return None
    if min(gaps) > dhi * 1.12:
        return None
    rng.shuffle(hosts_all)
    hosts = hosts_all[:T["H"]]
    hg = [hd_px(fam["combos"][h], q) for h in hosts for q in polys]
    if hg and min(hg) < dlo * 0.85:
        return None
    return cands, hosts, blocks, (min(gaps), max(gaps)), (min(hg) if hg else None)


# ------------------------------------------------------------ probe points
def probes(poly, marks):
    n = len(poly)
    ins = set()
    arcs, apex, shoulders = [], [], []
    for e in sorted(marks):
        ia, iap, ib = marks[e]
        ins |= {ia, iap, ib}
        Pp, Qq = poly[(ia - 1) % n], poly[(ib + 1) % n]
        shoulders.append(arr(seg_samples(*trim_toward(Pp, poly[ia],
                                                      SHOULDER), 1.0)))
        shoulders.append(arr(seg_samples(*trim_toward(Qq, poly[ib],
                                                      SHOULDER), 1.0)))
        a1 = seg_samples(poly[ia], poly[iap], 1.0)
        a2 = seg_samples(poly[iap], poly[ib], 1.0)
        arcs.append((arr(a1), arr(a2)))
        apex.append(arr(disc_samples(poly[iap], APEX_R)))
    keep = [poly[i] for i in range(n) if i not in ins]
    base_edges = []
    for i in range(len(keep)):
        a, b = keep[i], keep[(i + 1) % len(keep)]
        Lq = math.hypot(b[0] - a[0], b[1] - a[1])
        if Lq >= MIN_BASE_EDGE:
            base_edges.append(arr(seg_samples(a, b, 1.0)))
    return dict(arcs=arcs, apex=apex, shoulders=shoulders,
                base_edges=base_edges, outline=outline_samples(poly, 1.0))


# ------------------------------------------------------- item construction
def build_item(rng, tier, rej=None):
    T = TIERS[tier]
    bump("item_try")
    fam = build_family(rng, T)
    if fam is None:
        return None
    got = pick_sets(rng, fam, T)
    if got is None:
        return None
    cands, hosts, blocks, gaps, hostgap = got
    cand_polys_all = [fam["combos"][c] for c in cands]
    # PANEL FIT: the four candidates must fit their frames.
    panel_off = panel_offset(cand_polys_all)
    if panel_off is None:
        bump("panel_fit")
        if rej is not None:
            rej["panelfit"] = rej.get("panelfit", 0) + 1
        return None

    # ------------------------------------------------------------ SIZE LAW
    # SIZE LAW.  Every shape in the stack follows ONE size law, taken from the
    # family itself.  size_pool is the bounding box of each of the four
    # CANDIDATE words.  The answer ci is drawn uniformly over those same four
    # words, so the target's box is one uniform draw from the pool; every host
    # and every filler is put on an independent uniform draw from it, EXACTLY
    # (fit_box, not fit).  All n boxes on a board are then i.i.d. from one
    # four-atom distribution, so the target's size is exchangeable with every
    # other shape's -- and, because the atoms are hit exactly, shapes that draw
    # the same atom TIE, and a tie is broken by draw order, which the shuffle
    # below makes uniform.  (Matching only max(bbox) and leaving the other side
    # to a tolerance would make every non-target jitter around the atom while
    # the target sat exactly on it.)
    #
    # The pool is built from the candidate SET, which pick_sets() fixes before
    # ci exists, so nothing in it knows which shape is the target -- the same
    # identity-free requirement dref has in place().
    size_pool = [bbox_wh(q) for q in cand_polys_all]

    ci = rng.randrange(4)                      # answer drawn uniformly
    correct_key = cands[ci]
    correct = fam["combos"][correct_key]

    # Hosts are same-base combos, so their size law is already close to the
    # target's.  They are still put on a drawn atom, because leaving them off it
    # would put target+fillers on the four atoms and the hosts beside them:
    # boxes would then coincide only among target+fillers, so "pick a shape
    # whose box repeats" would exclude the H hosts.  It also makes them better
    # decoys: host and target boxes collide outright.
    host_polys = []
    for h in hosts:
        Wj, Hj = size_pool[rng.randrange(4)]
        q = fam["combos"][h]
        w1, h1 = bbox_wh(q)
        if aniso(Wj / Hj, w1 / h1) > ANISO_CAP:
            bump("host_aniso")
            return None
        host_polys.append(fit_box(q, Wj, Hj))
    hgs = [hd_px(hp, q) for hp in host_polys for q in cand_polys_all]
    if hgs and min(hgs) < T["dpx"][0] * 0.85:
        bump("host_gap_fail")
        return None
    hostgap = min(hgs) if hgs else None

    # DECOY PLANTING: every signature value is also planted on filler shapes
    # (at most 2 per filler), so "which notch shape / which value occurs
    # anywhere in the picture" carries zero information about the answer, and
    # there is no side channel of the form "value 2 only ever grows on a host".
    vals = []
    for l in range(L_LOCI):
        vals += [fam["variants"][l][j].clone() for j in range(K_VAL)]
    rng.shuffle(vals)
    n_fill = T["n"] - 1 - T["H"]
    if n_fill < 4:
        return None
    quota = [0] * n_fill
    for i in range(len(vals)):
        quota[i % n_fill] += 1
    n_sim = int(round(T["fsim"] * n_fill))
    # One draw from the size law per filler SLOT, not per attempt: a slot keeps
    # the box it drew while it retries, so the filler that is accepted is the
    # one that met the draw -- not a draw that was moved to meet a filler.
    fill_box = [size_pool[rng.randrange(4)] for _ in range(n_fill)]
    fills, tries, vi = [], 0, 0
    while len(fills) < n_fill and tries < 4000:
        tries += 1
        if len(fills) < n_sim:
            # SIMILAR-SILHOUETTE FILLER: vertex-jittered from the candidate
            #   base, so the outline reads alike and only the detail differs.
            #   A coarse whole-silhouette comparison can then rule out nothing,
            #   and no clean, obviously foreign silhouette sits in plain view.
            s0 = jitter_base(rng, fam["base"], rng.uniform(0.16, 0.24))
            if s0 is None:
                continue
            s0 = fit(s0)
        else:
            s0 = fit(polygons.sample_polygon(rng, n_vert=rng.randint(5, 7),
                                             radius=BASE_R))
        need = quota[len(fills)]
        W_k, H_k = fill_box[len(fills)]
        ok = True
        for j in range(need):
            ns = len(s0)
            lens = [math.hypot(s0[(i + 1) % ns][0] - s0[i][0],
                               s0[(i + 1) % ns][1] - s0[i][1])
                    for i in range(ns)]
            ce = [i for i in range(ns) if lens[i] >= MIN_DECOY_EDGE]
            if not ce:
                ok = False
                break
            s2 = apply_notch(s0, rng.choice(ce), vals[vi + j])
            if s2 is None:
                ok = False
                break
            s0 = fit(s2)
        if not ok:
            bump("fill_notch_fail")
            continue
        # SIZE LAW, filler side.  The cuts are made on a 208px-normalised shape
        # so MIN_DECOY_EDGE / MIN_NOTCH_W keep their meaning; only the finished
        # outline is put on the box this slot drew.  Matching max(bbox) alone
        # would leave the second size channel open -- the family base is an
        # 8-9-gon and a plain filler a 5-7-gon, so the target fills its box
        # rounder (min/max 0.88 vs 0.85) and bbox area still separates them.
        # The quarter turn is free (fillers are sampled at arbitrary
        # orientations) and roughly doubles the acceptance rate; what is left
        # after it must be under FILL_ANISO, so the stretch fit_box applies
        # never exceeds ~2px and the planted signature values stay copies.
        w1, h1 = bbox_wh(s0)
        a_t = W_k / H_k
        if aniso(w1 / h1, a_t) > aniso(h1 / w1, a_t):
            s0 = rot90(s0)
            w1, h1 = h1, w1
        if aniso(w1 / h1, a_t) > FILL_ANISO:
            bump("fill_aniso")
            continue
        s0 = fit_box(s0, W_k, H_k)
        # A filler may not be a near miss of any candidate
        if min(hd_fast(s0, q) for q in cand_polys_all) < T["dpx"][0] * 1.5:
            bump("fill_near_cand")
            continue
        vi += need
        fills.append(s0)
    if len(fills) < n_fill:
        bump("fill_fail")
        return None

    shapes = [correct] + host_polys + fills
    kinds = ["target"] + ["host"] * T["H"] + ["fill"] * n_fill
    keys = [correct_key] + list(hosts) + [None] * n_fill
    idx = list(range(len(shapes)))
    rng.shuffle(idx)
    shapes = [shapes[i] for i in idx]
    kinds = [kinds[i] for i in idx]
    keys = [keys[i] for i in idx]
    tk = kinds.index("target")

    # ABSENT: no wrong candidate may nearly coincide with a stack shape
    wrong = [fam["combos"][c] for c in cands if c != correct_key]
    for w in wrong:
        if min(hd_fast(w, s) for s in shapes) < T["dpx"][0] * 0.85:
            bump("absent_fail")
            return None


    pr = probes(correct, fam["marks"][correct_key])
    # For the placement symmetrisation: the signature-arc point sets of the H
    #   same-base decoy hosts.  They are isomorphic to the target's (same base,
    #   same locus edges).  At placement time the target sticks to a host's
    #   signature arc while non-targets stick to the target's, so both sides
    #   use the same geometry and the placements can be identically
    #   distributed (see PLACEMENT LAW in the module docstring).
    host_arcs = {}
    for i2, kd in enumerate(kinds):
        if kd != "host":
            continue
        pa = probes(shapes[i2], fam["marks"][keys[i2]])["arcs"]
        host_arcs[i2] = np.concatenate([np.concatenate(a) for a in pa])
    order = ["c0", "c1", "c2", "c3"]
    rng.shuffle(order)
    cand_map = {f"c{i}": fam["combos"][cands[i]] for i in range(4)}
    answer = "ABCD"[order.index(f"c{ci}")]
    correct_idx = order.index(f"c{ci}")
    bump("item_pass")
    return dict(shapes=shapes, kinds=kinds, target_k=tk, fam=fam,
                host_arcs=host_arcs,
                outl=[outline_samples(s2, 1.5) for s2 in shapes],
                cands=cands, hosts=hosts, blocks=blocks,
                correct_key=correct_key,
                correct_idx=correct_idx,
                cand_map=cand_map, order=order, answer=answer,
                cand_polys=[cand_map[k] for k in order],
                panel_off=panel_off,
                gaps=gaps, hostgap=hostgap, pr=pr, n_planted=len(vals))


# ---------------------------------------------------------------- placement
MASK_W = 2 * int(R_OCC) + 1


def rast(poly, off):
    img = Image.new("1", (STACK_W, OUT_H), 0)
    d = ImageDraw.Draw(img)
    xy = [(x + off[0], y + off[1]) for x, y in poly]
    d.line(xy + [xy[0]], fill=1, width=MASK_W, joint="curve")
    for x, y in xy:
        d.ellipse([x - R_OCC, y - R_OCC, x + R_OCC, y + R_OCC], fill=1)
    return np.array(img, dtype=np.uint8)


def _ix(pts, off):
    x = np.clip(np.rint(pts[:, 0] + off[0]).astype(int), 0, STACK_W - 1)
    y = np.clip(np.rint(pts[:, 1] + off[1]).astype(int), 0, OUT_H - 1)
    return y, x


def covered(C, mk, pts, off):
    """Covered by the R_OCC neighbourhood of ANOTHER shape = total count
    minus own count > 0."""
    y, x = _ix(pts, off)
    return (C[y, x].astype(np.int16) - mk[y, x].astype(np.int16)) > 0


def band_pen(v, lo, hi):
    return 0.0 if lo <= v <= hi else (lo - v if v < lo else v - hi)


def state_of(item, pos, masks, C, T):
    """One snapshot: per-shape visibility, the target's per-locus arc
    visibility, apex clearance, and the longest contiguous readable run."""
    tk = item["target_k"]
    off = pos[tk]
    vis = [float((~covered(C, masks[k], item["outl"][k], pos[k])).mean())
           for k in range(len(pos))]
    gvis = float((~covered(C, masks[tk], item["pr"]["outline"], off)).mean())
    arcv, apexv, runs = [], [], []
    for (a1, a2), ap in zip(item["pr"]["arcs"], item["pr"]["apex"]):
        v1 = ~covered(C, masks[tk], a1, off)
        v2 = ~covered(C, masks[tk], a2, off)
        arcv.append(float(np.concatenate([v1, v2]).mean()))
        runs.append(max(runs_of(v1)[1], runs_of(v2)[1]))
        apexv.append(float((~covered(C, masks[tk], ap, off)).mean()))
    return dict(vis=vis, gvis=gvis, arcv=arcv, apexv=apexv, runs=runs,
                m_arc=float(np.mean(arcv)), m_apex=float(np.mean(apexv)),
                max_vis=max(vis))


def block_stats(st, item):
    blocks = item["blocks"]
    vs = [max(st["arcv"][l] for l in bl) for bl in blocks]
    rs = [max(st["runs"][l] for l in bl) for bl in blocks]
    return min(vs), rs


def score(st, T, item):
    vstar, rs = block_stats(st, item)
    s = 0.0
    s += 8.0 * band_pen(vstar, *T["vstar"])
    s += 5.0 * band_pen(float(np.mean(st["arcv"])), *T["avis"])
    s += 3.0 * band_pen(st["gvis"], *T["rec"])
    s += 4.0 * max(0.0, st["max_vis"] - T["maxvis"])
    s += 0.30 * sum(1 for r in rs if r < T["chain_run"])
    return s


def place(rng, item, T, iters=70):
    """Random initial placement + aimed burial + hill-climb, until B* lands in
    this tier's band.

    THE PLACEMENT LAW LIVES ENTIRELY IN THIS FUNCTION.
      Every shape INCLUDING the target draws from the same free_pos(), and the
      annulus scale dref does not depend on which shape is the target.
      The aimed burial law stick_to() (place against someone else's outline or
      signature arc) is open to the target too, and the hill-climb has the
      target in its movable set.  Otherwise the target becomes the picture's
      one attractor and "the shape closest to the centroid" still fishes the
      answer out without reading a single notch.
    """
    shapes, tk = item["shapes"], item["target_k"]
    n = len(shapes)
    lo, hi = T["ov"]
    diams = [polygons.diameter(s2) for s2 in shapes]
    dref = float(np.mean(diams))       # the one annulus scale; identity-free
    cx, cy = STACK_W / 2, OUT_H / 2
    pos = [None] * n
    law = [None] * n
    base_focus = np.concatenate([np.concatenate(a)
                                 for a in item["pr"]["arcs"]])

    def inb(k, x, y):
        wk, hk = bbox_wh(shapes[k])
        return (wk / 2 + 8 < x < STACK_W - wk / 2 - 8 and
                hk / 2 + 8 < y < OUT_H - hk / 2 - 8)

    def free_pos(k):
        """The SHARED placement law -- the only one.  Target and non-targets
        draw from the same distribution at the same scale."""
        for _ in range(120):
            ang = rng.uniform(0, 2 * math.pi)
            r = (lo + (hi - lo) * rng.random() ** 1.7) * dref
            x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
            if inb(k, x, y):
                return (x, y)
        return None

    def stick_to(k, pts):
        """Aimed placement: put a point of shape k's outline onto a point of
        pts (+-4px jitter).  pts is another shape's sample set -- this is the
        same function for the target as for everybody else."""
        ol = item["outl"][k]
        for _ in range(80):
            q = ol[rng.randrange(len(ol))]
            pt = pts[rng.randrange(len(pts))]
            x = pt[0] - q[0] + rng.uniform(-4, 4)
            y = pt[1] - q[1] + rng.uniform(-4, 4)
            if inb(k, x, y):
                return (float(x), float(y))
        return None

    others = [k for k in range(n) if k != tk]
    hostk = sorted(item["host_arcs"])          # indices of same-base hosts

    def target_aim():
        """Aimed placement for the target: stick to some same-base decoy
        host's signature arc.  Geometrically identical to "a non-target sticks
        to the target's signature arc", which is why the two sides end up
        identically distributed."""
        if not hostk:
            return None
        j = hostk[rng.randrange(len(hostk))]
        return stick_to(tk, item["host_arcs"][j] + np.asarray(pos[j]))

    # (1) non-targets first, by the shared law
    for k in others:
        c = free_pos(k)
        if c is None:
            return None
        pos[k], law[k] = c, "shared"

    # (2) the target takes the SAME mixture law: with probability pfocus it
    #     sticks to a same-base host's signature arc, otherwise the shared
    #     annulus.  Identical to the mixture the non-targets get in (3).
    c, lb = None, "shared"
    if rng.random() < T["pfocus"]:
        c, lb = target_aim(), "aimed"
    if c is None:
        c, lb = free_pos(tk), "shared"
    if c is None:
        return None
    pos[tk], law[tk] = c, lb
    focus = base_focus + np.asarray(pos[tk])

    # (3) a pfocus-fraction of the non-targets is re-aimed at the target's
    #     signature arcs (aimed burial)
    for k in others:
        if rng.random() < T["pfocus"]:
            c = stick_to(k, focus)
            if c is not None:
                pos[k], law[k] = c, "aimed"

    masks = [rast(shapes[k], pos[k]) for k in range(n)]
    C = np.sum(masks, axis=0, dtype=np.int16)
    st = state_of(item, pos, masks, C, T)
    sc = score(st, T, item)
    for _ in range(iters):
        if sc <= 0.0:
            break
        k = rng.randrange(n)                       # target is movable too
        if k == tk:
            c, lb = None, "shared"
            if rng.random() < T["pfocus"]:
                c, lb = target_aim(), "aimed"
            if c is None:
                c, lb = free_pos(tk), "shared"
        else:
            vst, _rs = block_stats(st, item)
            need_more = (vst > T["vstar"][1] or
                         float(np.mean(st["arcv"])) > T["avis"][1] or
                         st["gvis"] > T["rec"][1])
            if need_more and rng.random() < 0.75:
                c, lb = stick_to(k, focus), "aimed"
            else:
                c, lb = free_pos(k), "shared"
        if c is None:
            continue
        mk = rast(shapes[k], c)
        C2 = C - masks[k] + mk
        old_pos, old_mask, old_law = pos[k], masks[k], law[k]
        pos[k], masks[k], law[k] = c, mk, lb
        st2 = state_of(item, pos, masks, C2, T)
        sc2 = score(st2, T, item)
        if sc2 < sc:
            C, st, sc = C2, st2, sc2
            if k == tk:
                focus = base_focus + np.asarray(pos[tk])
        else:
            pos[k], masks[k], law[k] = old_pos, old_mask, old_law
    if sc > 0.0:
        return None
    return pos, masks, C, st, dict(law=law, dref=dref, lo=lo, hi=hi)


# -------------------------- positional priors: probe family P_* and gate G_pos
# A positional prior is any rule of the form "the answer is the shape that is
# most <some positional extreme> in the picture".  Not one of them may beat
# chance 1/n, or the item can be solved without reading a single notch.
# Column 4 is "pos" for a positional prior (hard-gated by G_pos) or "size" for
# a size prior (reported, not gated: size is set by the scaling of candidate
# and filler shapes, and has nothing to do with placement; see the SIZE LAW
# block in build_item()).
POS_PROBES = (
    ("P_center",   "d_center",  "min", "pos"),   # closest to stack centre
    ("P_far",      "d_center",  "max", "pos"),   # farthest from the centre
    ("P_centroid", "d_centroid", "min", "pos"),  # closest to the centroid
    ("P_medoid",   "medoid",    "min", "pos"),   # min total dist to others
    ("P_top",      "cy",        "min", "pos"),   # topmost centre
    ("P_bottom",   "cy",        "max", "pos"),
    ("P_left",     "cx",        "min", "pos"),
    ("P_right",    "cx",        "max", "pos"),
    ("P_top_bb",   "by0",       "min", "pos"),   # topmost outline pixel
    ("P_left_bb",  "bx0",       "min", "pos"),
    ("P_bot_bb",   "by1",       "max", "pos"),
    ("P_right_bb", "bx1",       "max", "pos"),
    ("P_zfirst",   "zorder",    "min", "pos"),   # drawn first (bottom)
    ("P_zlast",    "zorder",    "max", "pos"),   # drawn last (top)
    ("P_big",      "barea",     "max", "size"),  # largest bounding box
    ("P_small",    "barea",     "min", "size"),
)


def shape_boxes(item, pos):
    out = []
    for k, sh in enumerate(item["shapes"]):
        xs = [x + pos[k][0] for x, y in sh]
        ys = [y + pos[k][1] for x, y in sh]
        out.append([round(min(xs), 2), round(min(ys), 2),
                    round(max(xs), 2), round(max(ys), 2)])
    return out


def pos_stats(item, pos):
    """One row per shape, position-only statistics."""
    B = np.asarray(shape_boxes(item, pos), float)
    ctr = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    cen = np.array([STACK_W / 2.0, OUT_H / 2.0])
    D = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=2)
    return dict(d_center=np.linalg.norm(ctr - cen, axis=1),
                d_centroid=np.linalg.norm(ctr - ctr.mean(0), axis=1),
                medoid=D.sum(1), cx=ctr[:, 0], cy=ctr[:, 1],
                bx0=B[:, 0], by0=B[:, 1], bx1=B[:, 2], by1=B[:, 3],
                barea=(B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]),
                zorder=np.arange(len(pos), dtype=float))


def pos_probe(item, pos):
    """Recorded once per item: for each positional prior, whether its pick is
    the target, plus the target's rank on that statistic.  Stored in the meta;
    the batch gate gate_pos_batch() consumes these directly."""
    S = pos_stats(item, pos)
    tk, n = item["target_k"], len(pos)
    out = {}
    for name, key, how, _cls in POS_PROBES:
        v = S[key]
        pick = int(np.argmin(v) if how == "min" else np.argmax(v))
        order = np.argsort(v if how == "min" else -v, kind="stable")
        out[name] = [int(pick == tk), int(np.where(order == tk)[0][0]), n]
    d = S["d_center"]
    out["d_center_target"] = round(float(d[tk]), 2)
    out["d_center_others_med"] = round(float(np.median(
        [d[k] for k in range(n) if k != tk])), 2)
    out["d_center_others"] = [round(float(d[k]), 2)
                              for k in range(n) if k != tk]
    return out


def pos_invariant(item, pos, plog):
    """The per-item half of gate G_pos: a structural invariant, NOT a filter.

    A positional prior can only come from the target taking a placement law
    nobody else takes.  So this does not check "where did the target rank on
    this board" -- filtering items by rank would only trade a positive prior
    for a negative one ("the answer is never the most central shape" is just as
    free), which is worse.  It checks the CONSTRUCTION:
      1. the target's placement label must be one of the shared laws
         (shared / aimed) -- the same two the other shapes may take, with no
         third law anywhere;
      2. when the label is "shared", the target's distance to the stack centre
         must lie inside the shared annulus [lo*dref, hi*dref] -- the same
         support and the same scale as every other shared shape;
      3. the annulus scale dref is the mean diameter over ALL shapes, so it
         does not depend on which shape is the target.
    A correct construction always passes; the moment somebody gives the target
    its own placement law again, this raises.  The other half of G_pos, "hit
    rate ~ 1/n", is judged on the batch by gate_pos_batch().
    """
    tk = item["target_k"]
    lb = plog["law"][tk]
    if lb not in ("shared", "aimed"):
        raise RuntimeError(f"G_pos: target placement law not in the shared "
                           f"set: {lb!r}")
    laws = set(plog["law"])
    if not laws <= {"shared", "aimed"}:
        raise RuntimeError(f"G_pos: unregistered placement law {laws}")
    if lb == "shared":
        r = math.hypot(pos[tk][0] - STACK_W / 2, pos[tk][1] - OUT_H / 2)
        blo, bhi = plog["lo"] * plog["dref"], plog["hi"] * plog["dref"]
        if not (blo - 1e-6 <= r <= bhi + 1e-6):
            raise RuntimeError(
                f"G_pos: target outside the shared annulus r={r:.2f} "
                f"band=[{blo:.2f},{bhi:.2f}]")
    return True


# ------------------------------------------------------------- cheap gates
def cheap_gates(item, pos, masks, C, st, T):
    n = len(pos)
    tk = item["target_k"]
    vis = st["vis"]
    if max(vis) > T["maxvis"]:
        return None
    if not (T["rec"][0] <= st["gvis"] <= T["rec"][1]):
        return None
    vstar, rs = block_stats(st, item)
    if not (T["vstar"][0] <= vstar <= T["vstar"][1]):
        return None
    if not (T["avis"][0] <= float(np.mean(st["arcv"])) <= T["avis"][1]):
        return None
    # CHAIN: every block must retain a contiguous readable arc of at least
    # chain_run px, or that block has no evidence left.  This is the machine
    # form of "a person has a chain of evidence they can walk".
    if min(rs) < T["chain_run"]:
        return None
    # adjacency / crossing relations
    adj = [[False] * n for _ in range(n)]
    for k in range(n):
        y, x = _ix(item["outl"][k], pos[k])
        for j in range(n):
            if j == k:
                continue
            if masks[j][y, x].any():
                adj[k][j] = adj[j][k] = True
    ncross = [sum(adj[k]) for k in range(n)]
    if min(ncross) < T["cross_min"]:
        return None
    seen, stk = {0}, [0]
    while stk:
        k = stk.pop()
        for j in range(n):
            if j not in seen and adj[k][j]:
                seen.add(j)
                stk.append(j)
    if len(seen) < n:
        return None
    rank = sorted(range(n), key=lambda k: vis[k]).index(tk)
    if rank == 0 or rank == n - 1:
        return None
    return dict(vis=[round(v, 3) for v in vis], ncross=ncross,
                global_vis=round(st["gvis"], 3),
                arc_vis=[round(v, 3) for v in st["arcv"]],
                mean_arc_vis=round(st["m_arc"], 3),
                Bstar=round(1 - st["m_arc"], 3),
                arc_vis_sorted=[round(v, 3) for v in sorted(st["arcv"])],
                Vstar=round(vstar, 3), block_runs=[int(r) for r in rs],
                blocks=[list(b) for b in item["blocks"]],
                apex_clear=[round(v, 3) for v in st["apexv"]],
                arc_runs=[int(r) for r in st["runs"]], occ_rank=rank,
                max_vis=round(max(vis), 3), min_ncross=min(ncross))


# ------------------ exhaustive-translation solver (the expensive gate)
def stack_ink(item, pos):
    """Render the DELIVERED image and threshold its stack region.  The gate
    must measure exactly the pixels an independent audit reads back from the
    PNG: one pixel of difference drops the clearance from sqrt(25)=5.0 to
    sqrt(20)=4.47 and the gate would leak items."""
    a = np.asarray(render(item, pos).convert("L"), dtype=np.uint8)
    out = np.zeros((OUT_H, STACK_W), bool)
    out[:, :STACK_W - 2] = a[:, :STACK_W - 2] < 170
    return out


def depth_map(ink):
    return distance_transform_edt(~ink)


SPAN = 24            # off-canvas margin for the full scan: going further out
                     # can only expose more white, never less


def uncovered_map(white, P):
    """For every integer translation, the number of outline points landing on
    white (ink does not explain them).  Off-canvas counts as white."""
    px = np.rint(P[:, 0]).astype(int)
    py = np.rint(P[:, 1]).astype(int)
    mnx, mny = px.min(), py.min()
    K = np.zeros((py.max() - mny + 1, px.max() - mnx + 1), np.float32)
    np.add.at(K, (py - mny, px - mnx), 1.0)
    SP = SPAN
    H, W = white.shape
    ip = np.ones((H + 2 * SP, W + 2 * SP), np.float32)
    ip[SP:SP + H, SP:SP + W] = white
    c = fftconvolve(ip, K[::-1, ::-1], mode="valid")
    return c, (mnx, mny, SP)


def local_minima(c, emin, slack=4.0, k=12, rad=6):
    cc = c.copy()
    out = []
    for _ in range(k):
        i = int(np.argmin(cc))
        v = float(cc.flat[i])
        if v > emin + slack:
            break
        iy, ix = divmod(i, cc.shape[1])
        out.append((iy, ix, v))
        y0, y1 = max(0, iy - rad), min(cc.shape[0], iy + rad + 1)
        x0, x1 = max(0, ix - rad), min(cc.shape[1], ix + rad + 1)
        cc[y0:y1, x0:x1] = np.inf
    return out


def at_offset(D, P, off, r):
    # Must round bit-for-bit the way uncovered_map does: rint(P) first, then
    # add the integer offset.  Writing rint(P+off) would, at .5 coordinates,
    # take numpy banker's rounding down a different branch and give a different
    # pixel, so the FFT scan's emin would not match what is recomputed here.
    xq = np.rint(P[:, 0]).astype(int) + int(round(off[0]))
    yq = np.rint(P[:, 1]).astype(int) + int(round(off[1]))
    dep = np.full(len(P), 1e6)
    ins = (xq >= 0) & (xq < D.shape[1]) & (yq >= 0) & (yq < D.shape[0])
    dep[ins] = D[yq[ins], xq[ins]]
    exposed = dep > r
    run = circ_max_run(exposed)
    mdep = float(dep[exposed].max()) if exposed.any() else 0.0
    return int(exposed.sum()), run, min(mdep, 1e5)


def host_offsets(item, pos):
    return [pos[k] for k in range(len(pos))
            if item["kinds"][k] in ("target", "host")]


def stageA(item, pos, D, T, r=R_READ):
    """Cheap prefilter: estimate the falsifying evidence only at the same-base
    host offsets (+-2px), which is where a wrong candidate is most likely to
    fit.  E* over all translations is <= this estimate, so failing here is
    already fatal and the FFT scan can be skipped."""
    ci = item["correct_idx"]
    hs = host_offsets(item, pos)
    est = []
    for i, poly in enumerate(item["cand_polys"]):
        if i == ci:
            continue
        P = outline_samples(poly, 1.0)
        best = (10 ** 9, 0, 0.0)
        for ox, oy in hs:
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    e, rn, dp = at_offset(D, P, (ox + dx, oy + dy), r)
                    if e < best[0]:
                        best = (e, rn, dp)
        if best[0] < T["estar"][0] or best[0] > T["estar"][1] * 2.0:
            return None
        if best[1] < RUN_FLOOR or best[2] < DEPTH_FLOOR:
            return None
        est.append(best)
    return est


def solve(item, pos, T, r=R_READ):
    """Exhaustive translation scan (FFT, every integer offset, including the
    SPAN margin of offsets that hang off the canvas)."""
    ink = stack_ink(item, pos)
    D = depth_map(ink)
    if stageA(item, pos, D, T, r) is None:
        return None
    white = D > r
    ci = item["correct_idx"]
    out = dict(emin=[], run=[], depth=[], nsite=[])
    for i, poly in enumerate(item["cand_polys"]):
        P = outline_samples(poly, 1.0)
        c, (mnx, mny, SP) = uncovered_map(white, P)
        emin = float(c.min())
        mins = local_minima(c, emin)
        worst_run, worst_dep = 10 ** 6, 10 ** 6
        for iy, ix, _v in mins:
            off = (ix - mnx - SP, iy - mny - SP)
            e2, rn, dp = at_offset(D, P, off, r)
            if e2 <= emin + 0.5:
                worst_run = min(worst_run, rn)
                worst_dep = min(worst_dep, dp)
        out["emin"].append(round(emin, 1))
        out["run"].append(int(worst_run if worst_run < 10 ** 6 else 0))
        out["depth"].append(round(worst_dep if worst_dep < 10 ** 5 else 0., 1))
        out["nsite"].append(len(mins))
        if i == ci:
            if emin > 0.5:
                return None
            iy, ix, _v = mins[0]
            off = np.array([ix - mnx - SP, iy - mny - SP], float)
            if np.hypot(*(off - np.asarray(pos[item["target_k"]]))) > 3.0:
                return None
        else:
            if emin < E_FLOOR:
                return None
            if not (T["estar"][0] <= emin <= T["estar"][1]):
                return None
            if out["run"][-1] < RUN_FLOOR:
                return None
            if out["depth"][-1] < DEPTH_FLOOR:
                return None
    wr = [out["emin"][i] for i in range(4) if i != ci]
    out["Estar"] = min(wr)
    out["correct_idx"] = ci
    return out


def rstar_of(item, pos, rmax=10):
    """The largest read tolerance r (px) under which the answer is still
    unique.  A GATE: make() refuses a board
    with r* < RSTAR_FLOOR (counted as rstarfail), and the value is written to
    meta.gates.rstar_px."""
    ink = stack_ink(item, pos)
    D = depth_map(ink)
    polys = item["cand_polys"]
    best = 0
    for r in range(1, rmax + 1):
        white = D > r
        zeros = 0
        for poly in polys:
            P = outline_samples(poly, 1.0)
            c, _ = uncovered_map(white, P)
            if c.min() < 0.5:
                zeros += 1
        if zeros == 1:
            best = r
        elif zeros > 1:
            break
    return best


_DVALS = np.unique(np.sqrt(np.add.outer(np.arange(33) ** 2,
                                        np.arange(33) ** 2).ravel().astype(float)))
_DVALS = _DVALS[_DVALS <= 32.0]   # every value a pixel-grid EDT takes, <= 32px

# render() draws at 2x and LANCZOS-halves, so polygon coordinate x lands on
# pixel index x - 0.5 (pixel i covers [i, i+1)); measured: ink-to-outline
# distances are smallest at a -0.5px shift.  The gap scan snaps outline samples
# to pixels at four sub-pixel phases, i.e. it tries every translation on a
# 0.5px lattice, so its verdict does not hinge on one rounding of the samples.
GAP_PHASES = ((-0.5, -0.5), (0.0, -0.5), (-0.5, 0.0), (0.0, 0.0))


def _gap_template(P):
    px = np.rint(P[:, 0]).astype(int)
    py = np.rint(P[:, 1]).astype(int)
    px -= px.min()
    py -= py.min()
    K = np.zeros((py.max() + 1, px.max() + 1), np.float64)
    np.add.at(K, (py, px), 1.0)
    return K[::-1, ::-1]


def _every_t_leaves_gap(D, Kf, th):
    """For every integer translation that keeps the snapped template inside D,
    is at least one sample at D >= th?  (A translation that pushes a sample out
    of the stack panel leaves that sample with no ink near it at all, so only
    the in-panel translations can fail.)"""
    if Kf.shape[0] > D.shape[0] or Kf.shape[1] > D.shape[1]:
        return True
    c = fftconvolve((D >= th).astype(np.float64), Kf, mode="valid")
    return float(c.min()) >= 0.5


def minimax_gap(D, P):
    """min over integer translations t of max over outline samples p of
    D[rint(p) + t]: how close the outline can be brought to the ink, measured
    at its WORST point.  D is the distance map of the stack panel (px to the
    nearest ink pixel).  Exact on the discrete set of EDT values (binary
    search; each probe is one FFT count of "samples with D >= th" over every
    translation).  Capped at 32px."""
    Kf = _gap_template(P)
    lo, hi = 0, len(_DVALS) - 1          # _DVALS[0] = 0: trivially true
    if _every_t_leaves_gap(D, Kf, _DVALS[hi]):
        return float(_DVALS[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _every_t_leaves_gap(D, Kf, _DVALS[mid]):
            lo = mid
        else:
            hi = mid
    return float(_DVALS[lo])


def stack_depth(item, pos):
    return depth_map(stack_ink(item, pos))


def gap_gate(item, pos, D=None, floor=None):
    """True iff every WRONG candidate, at every translation on the 0.5px
    lattice, keeps an outline point >= floor px from the stack ink.  12 FFTs;
    the cheap form of gap_margins() that make() calls first."""
    D = stack_depth(item, pos) if D is None else D
    floor = GAP_FLOOR if floor is None else floor
    for i, poly in enumerate(item["cand_polys"]):
        if i == item["correct_idx"]:
            continue
        S = outline_samples(poly, 0.5)
        for ph in GAP_PHASES:
            if not _every_t_leaves_gap(D, _gap_template(S + np.asarray(ph)),
                                       floor):
                return False
    return True


def gap_margins(item, pos, D=None):
    """minimax_gap() of every candidate (A-D order) against the DELIVERED stack
    ink, minimised over the four sub-pixel phases.  For the gold it is how
    well it fits (~0-1px); for a wrong candidate it is the white gap that
    remains at its most favourable translation.  0.5px outline samples, so a
    short spike cannot slip between samples."""
    D = stack_depth(item, pos) if D is None else D
    out = []
    for poly in item["cand_polys"]:
        S = outline_samples(poly, 0.5)
        out.append(min(minimax_gap(D, S + np.asarray(ph)) for ph in GAP_PHASES))
    return out


# --------------------------------------------------------------- rendering
def render(item, pos):
    W, H = OUT_W * SS, OUT_H * SS
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    sw = STROKE_OUT * SS
    for k, s in enumerate(item["shapes"]):
        xy = [((x + pos[k][0]) * SS, (y + pos[k][1]) * SS) for x, y in s]
        d.line(xy + [xy[0]], fill=(26, 26, 30), width=sw, joint="curve")
    d.line([(STACK_W * SS, 8 * SS), (STACK_W * SS, (OUT_H - 8) * SS)],
           fill=(210, 210, 216), width=2 * SS)
    f = ImageFont.load_default(20 * SS)     # Pillow's scalable default font (Pillow >= 10.1)
    for i, poly in enumerate(item["cand_polys"]):
        r, c = divmod(i, 2)
        x0, y0 = STACK_W + c * CELL + PAD, r * CELL + PAD
        box = (x0 * SS, y0 * SS, (x0 + CELL - 2 * PAD - 2) * SS,
               (y0 + CELL - 2 * PAD - 2) * SS)
        d.rounded_rectangle(box, radius=8 * SS, outline=(210, 210, 216),
                            width=2 * SS)
        d.text((box[0] + 8 * SS, box[1] + 5 * SS), "ABCD"[i],
               fill=(40, 40, 44), font=f)
        if item.get("panel_off") is not None:     # the item's registration
            ccx = box[0] + item["panel_off"][0] * SS
            ccy = box[1] + item["panel_off"][1] * SS
        else:                                     # no registration: frame centre
            ccx, ccy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 7 * SS
        xy = [(x * SS + ccx, y * SS + ccy) for x, y in poly]
        d.line(xy + [xy[0]], fill=(26, 26, 30), width=sw, joint="curve")
    return img.resize((OUT_W, OUT_H), Image.LANCZOS)


# ------------------------------------------------------------- main pipeline
def make(seed, idx, tier, want_rstar=True):
    T = TIERS[tier]
    rng = polygons.derive_rng(f"{_DIAL_STREAM}:{seed}:{tier}", idx)
    rej = dict(geom=0, place=0, cheap=0, solve=0, gapfail=0, rstarfail=0,
               panelfit=0)        # panelfit is a subset of geom
    for _g in range(120):
        item = build_item(rng, tier, rej)
        if item is None:
            rej["geom"] += 1
            continue
        for _p in range(140):
            bump("place_try")
            got = place(rng, item, T)
            if got is None:
                rej["place"] += 1
                continue
            pos, masks, C, st, plog = got
            pos_invariant(item, pos, plog)      # G_pos per-item invariant
            gg = cheap_gates(item, pos, masks, C, st, T)
            if gg is None:
                rej["cheap"] += 1
                continue
            bump("cheap_pass")
            sv = solve(item, pos, T)
            if sv is None:
                rej["solve"] += 1
                continue
            bump("solve_pass")
            # UNIQUENESS MARGIN: translation-uniform gap, then r*
            Dst = stack_depth(item, pos)
            if not gap_gate(item, pos, Dst):
                rej["gapfail"] += 1
                bump("gap_fail")
                continue
            rs = rstar_of(item, pos) if want_rstar else None
            if rs is not None and rs < RSTAR_FLOOR:
                rej["rstarfail"] += 1
                bump("rstar_fail")
                continue
            bump("margin_pass")
            gm = gap_margins(item, pos, Dst)
            fam = item["fam"]
            meta = dict(
                tier=tier, canvas=[OUT_W, OUT_H], n_shapes=T["n"],
                n_loci=L_LOCI, n_hosts=T["H"], n_planted=item["n_planted"],
                answer=item["answer"],
                shape_bbox_px=round(max(bbox_wh(item["cand_polys"][0])), 1),
                cand_gap_px=[round(item["gaps"][0], 1),
                             round(item["gaps"][1], 1)],
                host_gap_px=(round(item["hostgap"], 1)
                             if item["hostgap"] else None),
                occlusion_frac=round(1 - gg["global_vis"], 3),
                Estar_px=sv["Estar"], emin_all=sv["emin"],
                refute_run_px=sv["run"], refute_depth_px=sv["depth"],
                n_sites=sv["nsite"], **gg, pos_probe=pos_probe(item, pos),
                pos_law=plog["law"], pos_dref=round(plog["dref"], 2), **rej,
                cand_keys=[list(k) for k in item["cands"]],
                block_map=[list(b) for b in item["blocks"]],
                host_keys=[list(k) for k in item["hosts"]],
                correct_key=list(item["correct_key"]),
                loci=[f"edge{e}: v0={item['fam']['variants'][l][0].name()}"
                      f" | v1={item['fam']['variants'][l][1].name()}"
                      for l, e in enumerate(fam["edges"])])
            if want_rstar:
                meta["rstar_px"] = rs
            meta["gap_margin_px"] = [round(g, 3) for g in gm]
            meta["panel_off"] = [float(v) for v in item["panel_off"]]
            return item, pos, meta
        rej["geom"] += 1
    return None
