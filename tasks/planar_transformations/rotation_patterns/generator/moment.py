# -*- coding: utf-8 -*-
"""Ink-multipole shortcuts, and how far a word can push them down.

A candidate IS the base rotated by 360*s/16, so EVERY rotation-covariant global
statistic of the ink rotates with it.  For the m-th multipole
    A_m(w) = sum_j q_m(level of spoke j) * exp(i*m*theta_j),  theta_j = j*2pi/16
a rotation by s steps multiplies A_m by exp(i*m*2pi*s/16).  Read its phase in
the base and in a candidate and you get s mod 16/gcd(m,16) - for free, with no
reading of the code at all.  m=1 is the ink centroid; m=2 is the principal axis
of the ink covariance.  The hub disc, the rim ring and the grey guide circles
are concentric, so they contribute 0 to every m>0 and cannot help.

How far down can a WORD push A_m?

  m=1.  A_1 = sum_j q(w_j) zeta^j with zeta a primitive 16th root.  Writing
        zeta^{j+8} = -zeta^j, A_1 = sum_{j<8} (q(w_j)-q(w_{j+8})) zeta^j, and
        {zeta^0..zeta^7} is an R-basis only in the sense of 2 real equations on
        8 real unknowns - but the unknowns live in the DISCRETE set of level
        differences.  For equally spaced q's the coefficients are integer
        multiples of one unit, {1,zeta,..,zeta^7} is a Z-basis of Z[zeta_16], so
        A_1 = 0 forces q(w_j) = q(w_{j+8}) for all j, i.e. period 8, i.e. a
        NON-PRIMITIVE word whose candidates collide.  Exact cancellation is
        therefore impossible; the best a word can do is make |A_1| small.  It
        gets small enough: the floor over binary words is |A_1| = 0.082 units
        against a median of 1.78, and at that level the measured centroid offset
        (0.10 delivered px) is buried in rasterisation noise - the probe drops to
        chance.  So the pools filter the WORD rather than bend the symbol.

  m=2.  zeta^{2j} has period 8, so A_2 = sum_{r<4} (b_r - b_{r+4}) omega^r with
        b_r = q2(w_r)+q2(w_{r+8}) and omega = exp(i*pi/4).  Because 1/sqrt(2) is
        irrational, A_2 = 0 has an EXACT combinatorial solution:
              q2(w_r) + q2(w_{r+8}) == q2(w_{r+4}) + q2(w_{r+12}),  r = 0..3
        and that does NOT force periodicity.  So m=2 is killed exactly, not
        merely shrunk.  (The easy pools do not apply this gate; see build/exhaustive_pools.py.)

  m=3.  gcd(3,16)=1 so it pins s outright like m=1, and like m=1 it can only be
        shrunk.  Reported, and filtered on the same footing as m=1.
"""
from __future__ import annotations
import cmath, math
from . import dial as D
from .word import K

_Z = [cmath.exp(2j * math.pi * j / K) for j in range(K)]


def q_of(p, m):
    """The m-th radial weight of ONE spoke (bar + bead), in closed form."""
    if m == 1:
        return D.W_BAR * (p ** 2 - D.R_IN ** 2) / 2.0 + math.pi * D.R_BEAD ** 2 * p
    if m == 2:
        return D.W_BAR * (p ** 3 - D.R_IN ** 3) / 3.0 + math.pi * D.R_BEAD ** 2 * p ** 2
    if m == 3:
        return D.W_BAR * (p ** 4 - D.R_IN ** 4) / 4.0 + math.pi * D.R_BEAD ** 2 * p ** 3
    raise ValueError(m)


def centroid_px(w, L):
    """Ideal (un-rasterised) centroid offset of the figure, delivered px."""
    lv = D.levels(L)
    v = sum(q_of(lv[c], 1) * _Z[j] for j, c in enumerate(w))
    mass = (math.pi * D.R_HUB ** 2 + 2 * math.pi * D.R_RIM * D.W_RIM
            + sum(D.ray_mass(lv[c]) for c in w))
    return abs(v) / mass
