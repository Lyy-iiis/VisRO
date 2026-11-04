"""Bitmask exact-cover tiler and the target sampler of the rigidity dial.

A search node is one
attempted placement (piece, pose, translation) that covers the current anchor
cell and fits inside the remaining region.  For a NON-tiling set the search is
exhaustive, so the whole tree is explored and the node count is independent of
the order in which pieces/poses are tried.  Only solve_nodes (early exit)
depends on ordering.
"""
from __future__ import annotations

from .base import generate as G


# --------------------------------------------------------------------------- #
# poses / signature
# --------------------------------------------------------------------------- #
def mirror(cells):
    return G.normalize((-x, y) for x, y in cells)


def poses(cells, rotations=True, flips=False):
    out = []
    for base in (cells, mirror(cells)) if flips else (cells,):
        for o in G.orientations(base, rotations):
            if o not in out:
                out.append(o)
    return out


def sig(p):
    """(cell count, perimeter, sorted bbox dims)."""
    s = set(p)
    per = 0
    for (x, y) in s:
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb not in s:
                per += 1
    xs = [x for x, _ in s]
    ys = [y for _, y in s]
    return (len(s), per, tuple(sorted((max(xs) - min(xs) + 1,
                                       max(ys) - min(ys) + 1))))


# --------------------------------------------------------------------------- #
# bitmask machinery
# --------------------------------------------------------------------------- #
def index_target(target):
    """cell -> bit index, ordered by (y, x) so 'lowest set bit' == the
    anchor rule min(remaining, key=(y, x))."""
    cells = sorted(target, key=lambda c: (c[1], c[0]))
    return {c: i for i, c in enumerate(cells)}, cells


def placements(target, part, idx, rotations=True, flips=False):
    """-> dict anchor_bit_index -> list of masks (each covering that cell)."""
    tset = set(target)
    by_anchor = {}
    for pose in poses(part, rotations, flips):
        pose_cells = sorted(pose)
        for (ax, ay) in tset:
            for (cx, cy) in pose_cells:
                dx, dy = ax - cx, ay - cy
                trans = [(x + dx, y + dy) for x, y in pose_cells]
                if not all(c in tset for c in trans):
                    continue
                m = 0
                for c in trans:
                    m |= 1 << idx[c]
                by_anchor.setdefault(idx[(ax, ay)], set()).add(m)
    return {a: sorted(v) for a, v in by_anchor.items()}


def tile_search(target, parts, rotations=True, flips=False, cap=4_000_000):
    """-> (solved, nodes, capped)."""
    target = G.normalize(target)
    if sum(len(p) for p in parts) != len(target):
        return False, 0, False
    idx, cells = index_target(target)
    n = len(cells)
    full = (1 << n) - 1
    tabs = [placements(target, p, idx, rotations, flips) for p in parts]
    state = {"n": 0, "cap": False}
    npar = len(parts)

    def lowbit_index(m):
        return (m & -m).bit_length() - 1

    def solve(remaining, unused):
        if not remaining:
            return True
        a = lowbit_index(remaining)
        for k in range(npar):
            if not (unused >> k) & 1:
                continue
            for m in tabs[k].get(a, ()):
                if m & ~remaining:
                    continue
                state["n"] += 1
                if state["n"] > cap:
                    state["cap"] = True
                    return False
                if solve(remaining & ~m, unused & ~(1 << k)):
                    return True
                if state["cap"]:
                    return False
        return False

    ok = solve(full, (1 << npar) - 1)
    return ok, state["n"], state["cap"]


# --------------------------------------------------------------------------- #
# THE DIAL: target = bounding rectangle with k cells bitten off the boundary
# --------------------------------------------------------------------------- #
def bite_rect(rng, w, h, k, tries=300):
    """W x H rectangle with exactly k boundary cells removed, still connected
    and still spanning the full W x H bounding box (so the option panels can
    never be scored against a smaller/other rectangle).  k = 0 -> solid."""
    full = {(x, y) for x in range(w) for y in range(h)}
    for _ in range(tries):
        cur = set(full)
        ok = True
        for _ in range(k):
            cand = [c for c in cur
                    if any((c[0] + dx, c[1] + dy) not in cur
                           for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
            rng.shuffle(cand)
            done = False
            for c in cand:
                rest = cur - {c}
                if not G.connected(rest):
                    continue
                xs = [p[0] for p in rest]
                ys = [p[1] for p in rest]
                if (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1) != (w, h):
                    continue
                cur = rest
                done = True
                break
            if not done:
                ok = False
                break
        if ok and len(cur) == w * h - k:
            return G.normalize(frozenset(cur))
    return None
