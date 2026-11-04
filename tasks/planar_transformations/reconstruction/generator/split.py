"""Piece sizes and the carving sampler of reconstruction.

grow_split carves a target into connected regions of the given sizes, keeping
the not-yet-carved remainder connected at every step, so the sampler succeeds
instead of painting itself into a corner.  The true pieces and every decoy
carving come from it (decoys.carvings).
"""
from __future__ import annotations


def sample_sizes(rng, total, n, lo, hi):
    sizes = [lo] * n
    left = total - lo * n
    if left < 0:
        return None
    while left:
        k = rng.randrange(n)
        if sizes[k] < hi:
            sizes[k] += 1
            left -= 1
    return sizes


def _bit_index(target):
    """cell -> bit on a grid with one guard column (stride = width + 1), so a
    +-1 shift never wraps from one row into the next.  -> (bits, stride)"""
    x0 = min(x for x, _ in target)
    y0 = min(y for _, y in target)
    stride = max(x for x, _ in target) - x0 + 2
    return {(x, y): 1 << ((y - y0) * stride + (x - x0)) for x, y in target}, stride


def _bits_connected(m, stride):
    """4-connectivity of the cell set encoded in bitmask m (empty -> True):
    flood fill from the lowest cell by shifting, the same answer as
    generate.connected on the cells, without building sets."""
    if not m:
        return True
    f = m & -m
    while True:
        g = (f | (f << 1) | (f >> 1) | (f << stride) | (f >> stride)) & m
        if g == f:
            return f == m
        f = g


def grow_split(rng, target, sizes, tries=30):
    """Carve `sizes` connected regions out of `target`, keeping the not-yet
    carved remainder connected at every step (which is what makes the sampler
    succeed instead of painting itself into a corner).

    The connectivity tests run on bitmasks (_bits_connected) instead of
    generate.connected on sets: same answers, same rng calls, same output,
    about 4x less time (the tests are ~70% of an xhard/xxhard build)."""
    order = list(sizes)
    bit, stride = _bit_index(target)
    full = 0
    for v in bit.values():
        full |= v
    for _ in range(tries):
        rng.shuffle(order)
        rest = set(target)
        rest_m = full
        parts = []
        good = True
        for si, s in enumerate(order):
            if si == len(order) - 1:
                if len(rest) == s and _bits_connected(rest_m, stride):
                    parts.append(frozenset(rest))
                else:
                    good = False
                break
            key = rng.choice([lambda c: (c[1], c[0]), lambda c: (-c[1], c[0]),
                              lambda c: (c[0], c[1]), lambda c: (-c[0], c[1])])
            first = min(rest, key=key)
            reg = {first}
            reg_m = bit[first]
            while len(reg) < s:
                fr = sorted({nb for (x, y) in reg
                             for nb in ((x + 1, y), (x - 1, y),
                                        (x, y + 1), (x, y - 1))
                             if nb in rest and nb not in reg})
                if not fr:
                    good = False
                    break
                rng.shuffle(fr)
                pick = None
                left = rest_m & ~reg_m
                for c in fr:                       # prefer non-fragmenting
                    if _bits_connected(left & ~bit[c], stride):
                        pick = c
                        break
                if pick is None:
                    pick = fr[0]
                reg.add(pick)
                reg_m |= bit[pick]
            if not good:
                break
            if not _bits_connected(rest_m & ~reg_m, stride):
                good = False
                break
            parts.append(frozenset(reg))
            rest -= reg
            rest_m &= ~reg_m
        if good and len(parts) == len(order) and \
           sorted(len(p) for p in parts) == sorted(sizes):
            return parts
    return None

