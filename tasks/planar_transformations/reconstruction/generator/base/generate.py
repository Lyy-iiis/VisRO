"""Polyomino helpers of reconstruction: normalisation, rotations, canonical forms, connectivity.

canon(cells, rotations) is the minimum over the allowed orientations of the
normalised cell set; set_key(parts, rotations) is the multiset of canonical part
shapes, used to deduplicate option sets.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Polyomino helpers (cells = frozenset of (x, y))
# --------------------------------------------------------------------------- #
def normalize(cells):
    cells = list(cells)
    x0 = min(x for x, _ in cells)
    y0 = min(y for _, y in cells)
    return frozenset((x - x0, y - y0) for x, y in cells)


def rot90(cells):
    return normalize((y, -x) for x, y in cells)


def orientations(cells, rotations):
    cells = normalize(cells)
    if not rotations:
        return [cells]
    out = []
    c = cells
    for _ in range(4):
        if c not in out:
            out.append(c)
        c = rot90(c)
    return out


def canon(cells, rotations):
    return min(orientations(cells, rotations),
               key=lambda s: sorted(s))


def connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        x, y = c
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


# --------------------------------------------------------------------------- #
# Problem building
# --------------------------------------------------------------------------- #
def set_key(parts, rotations):
    return tuple(sorted(tuple(sorted(canon(p, rotations))) for p in parts))
