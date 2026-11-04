"""Drawing kernel: the dial, with L spoke LENGTHS.

ONE SOLID SPOKE per direction, running outward from the central hub, ending in
a round bead.  How far out the bead sits IS the code (one of L radii).  No
break, no second piece, no mirror.

Aids that carry no information (they are concentric, hence identical in every
figure and unchanged by any rotation):
  * L faint grey guide circles, one at each level radius, so "how long is this
    spoke" is read as "which circle is this bead sitting on";
  * a hub disc and an outer rim ring, both drawn in the SAME colour as the
    spokes so that no colour channel can separate them from the code.

k = 16 spokes on the exact 16-fold angular lattice, one pointing straight up in
every figure => the set of spoke DIRECTIONS is pixel-identical in the base and
in all candidates.  Bearing entropy of the lattice is 0.
"""
from __future__ import annotations
import math, itertools
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from shared.drawing.fonts import load_bundled_font

from .word import K

BG = (255, 255, 255)
BOX = 168                # delivered side of one figure
SS = 6                   # supersample factor (draw at 1008, LANCZOS to 168)
TILE = BOX * SS
UP = -math.pi / 2        # j=0 points straight up; clockwise positive (PIL y down)

R_HUB = 26.0             # hub disc, same colour as the spokes
R_IN = 26.0              # every bar starts here
P_MIN = 34.0             # innermost level radius (bead centre)
P_MAX = 67.0             # outermost level radius
W_BAR = 3.0              # bar width, constant for every level
R_BEAD = 5.0             # bead radius, constant for every level
R_RIM = 79.6             # rim ring centre-line, same colour
W_RIM = 4.6
RING = (203, 203, 209)   # grey guide circles
W_RING = 1.3

PALETTE = [
    (198, 40, 40), (21, 101, 192), (27, 122, 62), (106, 27, 154),
    (191, 96, 0), (0, 121, 124), (173, 20, 87), (55, 71, 130),
]


def font_num(sz):
    """The bundled DejaVu Sans Bold that rasterises the index badges; a
    different font file changes the badge pixels."""
    return load_bundled_font(sz, "sans-bold")


def levels(L):
    """The L bead radii, equally spaced between the SAME two rails at every tier."""
    if L == 1:
        return [P_MAX]
    st = (P_MAX - P_MIN) / (L - 1)
    return [P_MIN + v * st for v in range(L)]


def level_step(L):
    return (P_MAX - P_MIN) / (L - 1)


# ---- the ink mass of one spoke, in closed form (used by the word gate W8)
def ray_mass(p):
    """ink area of one spoke = bar + bead."""
    return W_BAR * (p - R_IN) + math.pi * R_BEAD ** 2


# ------------------------------------------------------------------- drawing
def _bar(d, cx, cy, s, ang, r0, r1, wid, col):
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    h = wid / 2.0
    d.polygon([(cx + s * (r0 * ux + h * nx), cy + s * (r0 * uy + h * ny)),
               (cx + s * (r1 * ux + h * nx), cy + s * (r1 * uy + h * ny)),
               (cx + s * (r1 * ux - h * nx), cy + s * (r1 * uy - h * ny)),
               (cx + s * (r0 * ux - h * nx), cy + s * (r0 * uy - h * ny))], fill=col)


def _ring(d, cx, cy, s, r, wid, col):
    d.ellipse([cx - r * s, cy - r * s, cx + r * s, cy + r * s],
              outline=col, width=max(1, int(round(wid * s))))


def draw_dial(d, cx, cy, s, w, L, col):
    lv = levels(L)
    for p in lv:                                   # grey guide circles
        _ring(d, cx, cy, s, p, W_RING, RING)
    for j, c in enumerate(w):                      # the spokes
        a = UP + 2 * math.pi * j / K
        p = lv[c]
        _bar(d, cx, cy, s, a, R_IN, p, W_BAR, col)
        bx, by = cx + s * p * math.cos(a), cy + s * p * math.sin(a)
        r = R_BEAD * s
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=col)
    _ring(d, cx, cy, s, R_RIM, W_RIM, col)         # rim, same colour
    d.ellipse([cx - R_HUB * s, cy - R_HUB * s, cx + R_HUB * s, cy + R_HUB * s], fill=col)


def render_tile(w, L, col):
    img = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    draw_dial(ImageDraw.Draw(img), TILE / 2.0, TILE / 2.0, SS, w, L, col)
    return img


def rotate_tile(t, cw_deg):
    return t.rotate(-cw_deg, resample=Image.BICUBIC, expand=False,
                    center=(t.width / 2, t.height / 2))


def flatten(t, box=BOX):
    im = Image.new("RGB", (box, box), BG)
    r = t.resize((box, box), Image.LANCZOS)
    im.paste(r, (0, 0), r)
    return np.asarray(im).astype(np.int16)


# --------------------------------------------------------------- pixel probes
def color_ink(a):
    """Saturation weight: the grey guide circles score ~0, the coloured dial high.
    This is the STRICT ink for the centroid probe - it cannot be diluted by any
    grey furniture we add."""
    f = a.astype(np.float64)
    return f.max(axis=2) - f.min(axis=2)


def centroid_of(ink):
    tot = ink.sum()
    n = ink.shape[0]
    c = (n - 1) / 2.0
    if tot <= 0:
        return 0.0, 0.0
    ys, xs = np.mgrid[0:n, 0:n]
    return (math.hypot((ink * xs).sum() / tot - c, (ink * ys).sum() / tot - c),
            math.atan2((ink * ys).sum() / tot - c, (ink * xs).sum() / tot - c))


def worst_cc(arrs, box=BOX, floor=None):
    worst = 10 ** 9
    for i, j in itertools.combinations(range(len(arrs)), 2):
        dm = np.abs(arrs[i] - arrs[j]).max(axis=2) > 45
        if not dm.any():
            return 0
        lab, n = ndimage.label(dm)
        worst = min(worst, int(ndimage.sum(dm, lab, range(1, n + 1)).max()))
        if floor is not None and worst < floor:
            return worst
    return worst


def rot_fidelity(base_img, arrs, shifts, box=BOX):
    """G5: the directly drawn candidate must equal the base bitmap actually
    rotated by 360*s/16 - the figures really are rotations of one another."""
    worst = 0.0
    for direct, s in zip(arrs, shifts):
        rotd = flatten(rotate_tile(base_img, 360.0 * s / K), box)
        worst = max(worst, float((np.abs(direct - rotd).max(axis=2) > 45).mean()))
    return worst


def _one_symbol(L, v, col, box=BOX):
    img = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = levels(L)[v]
    _bar(d, TILE / 2.0, TILE / 2.0, SS, UP, R_IN, p, W_BAR, col)
    bx, by = TILE / 2.0, TILE / 2.0 + SS * p * math.sin(UP)
    r = R_BEAD * SS
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=col)
    return flatten(img, box)


def level_legibility(L, col, box=BOX):
    """Adjacent-level separability, measured on DELIVERED pixels."""
    lv = levels(L)
    out = []
    for v in range(L - 1):
        a0, a1 = _one_symbol(L, v, col, box), _one_symbol(L, v + 1, col, box)
        dm = np.abs(a0 - a1).max(axis=2) > 45
        lab, n = ndimage.label(dm)
        blob = int(ndimage.sum(dm, lab, range(1, n + 1)).max()) if n else 0
        out.append(dict(v=v, d_rad=lv[v + 1] - lv[v], blob=blob, xor=int(dm.sum()),
                        bead_gap=(lv[v + 1] - lv[v]) - 2 * R_BEAD))
    return out


def geom():
    return dict(
        step_L2=level_step(2), step_L3=level_step(3), step_L4=level_step(4),
        step_L5=level_step(5),
        bead_pitch_min=2 * math.pi * P_MIN / K - 2 * R_BEAD,
        bar_pitch_hub=2 * math.pi * R_IN / K - W_BAR,
        rim_clear=(R_RIM - W_RIM / 2) - (P_MAX + R_BEAD),
        tile_margin=BOX / 2.0 - (R_RIM + W_RIM / 2),
        hub_to_first=P_MIN - R_BEAD - R_HUB,
    )
