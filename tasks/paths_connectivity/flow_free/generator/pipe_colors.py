"""flow_free pipe colours and the colour metric that keeps them apart."""

from __future__ import annotations

import math


PIPE_COLORS = [
    ("red",     (227, 38, 38)),
    ("blue",    (40, 90, 230)),
    ("green",   (35, 175, 55)),
    ("yellow",  (240, 215, 30)),
    ("orange",  (240, 140, 40)),
    ("purple",  (175, 55, 215)),
    ("cyan",    (40, 200, 215)),
    ("pink",    (240, 110, 200)),
    ("white",   (245, 245, 245)),
    ("magenta", (220, 40, 140)),
    ("lime",    (150, 220, 40)),
    ("brown",   (150, 90, 50)),
]


# ---------------------------------------------------------------- colour separation
#
# A ringed one-pipe question asks which of two same-coloured runs the ring sits
# on. That judgement is only clean if pipes of DIFFERENT colours are obviously
# different. The full 12-colour palette is not: CIE76 dE between
# the closest pairs is pink/magenta 22.9, green/lime 32.4, blue/purple 38.6.
# On a cluttered board those read as "maybe the same colour", which would put
# perceptual difficulty back in through the side door.
#
# Max-clique over the palette at several thresholds:
#     dE>=40  9 colours    dE>=55  6 colours    dE>=70  5 colours
# 55 is the tightest threshold that still leaves 6 to choose from, which is one
# more than the 5 distinct colours a 6-pipe board needs -- so boards still vary
# instead of always using one fixed set.
DE_MIN = 55.0


def lab(rgb):
    r, g, b = [v / 255 for v in rgb]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    x = (r * .4124 + g * .3576 + b * .1805) / .95047
    y = r * .2126 + g * .7152 + b * .0722
    z = (r * .0193 + g * .1192 + b * .9505) / 1.08883
    h = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    x, y, z = h(x), h(y), h(z)
    return (116 * y - 16, 500 * (x - y), 200 * (y - z))


_LAB = {name: lab(rgb) for name, rgb in PIPE_COLORS}


def delta_e(a, b):
    la, lb = _LAB[a[0]], _LAB[b[0]]
    return math.dist(la, lb)


# The ring is drawn white, so the TARGET's colour must be distinguishable from
# white too -- otherwise the marker is invisible on the one pipe it marks.
# dE(white, cyan)=44.6, so cyan is out as well; both are caught by reusing DE_MIN.
RING_RGB = (255, 255, 255)
_RING_LAB = lab(RING_RGB)


def ring_visible_on(colour):
    return math.dist(_LAB[colour[0]], _RING_LAB) >= DE_MIN
