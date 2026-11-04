"""flow_free Backbite boards: Hamiltonian paths that never fall back to a fixed board.

A randomized depth-first search for a Hamiltonian path needs a node budget and a
fallback for when the budget runs out (the canonical boustrophedon "snake"), and
a tortuosity bias makes that fallback MORE likely: the harder the tier, the more
often the board would collapse to the single easiest board in the space.

Backbite (Mansfield's move) instead starts from the snake and repeatedly takes
an end of the path and a uniformly random grid-neighbour q of that end; if q is
not already the path-adjacent cell, q = p_j, and reversing the sub-path beyond j
yields another Hamiltonian path.  The move can never fail, so there is no
fallback.  k = 6*n*n moves.

Tortuosity is a real knob: the move is accepted by Metropolis-Hastings on the
turn count, energy = -beta * turns, with the proposal correction
deg(old_end)/deg(new_end).  beta is a monotone function of `tort`.  Because the
chain starts AT the snake, a board that is still one of the 16 snake images
(identity, 8 dihedral images, each forward and reversed) is rejected and the
chain re-run with more moves.

cut_into_segments then cuts the path into the board's pipes: contiguous,
non-crossing segments that together cover every cell.
"""
from __future__ import annotations

import math


# --------------------------------------------------------------------- backbite
def _snake(n):
    p = []
    for r in range(n):
        p.extend((r, c) for c in (range(n) if r % 2 == 0 else range(n - 1, -1, -1)))
    return p


def _nbrs(r, c, n):
    if r > 0: yield (r - 1, c)
    if r < n - 1: yield (r + 1, c)
    if c > 0: yield (r, c - 1)
    if c < n - 1: yield (r, c + 1)


def _degree(cell, n):
    return sum(1 for _ in _nbrs(cell[0], cell[1], n))


def _turns(path):
    return sum(1 for a, b, c in zip(path, path[1:], path[2:])
               if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]))


def turn_frac(path):
    return _turns(path) / (len(path) - 2) if len(path) > 2 else 0.0


def _snake_images(n):
    base = _snake(n)
    out = set()
    for f in (lambda r, c: (r, c), lambda r, c: (r, n - 1 - c),
              lambda r, c: (n - 1 - r, c), lambda r, c: (n - 1 - r, n - 1 - c),
              lambda r, c: (c, r), lambda r, c: (c, n - 1 - r),
              lambda r, c: (n - 1 - c, r), lambda r, c: (n - 1 - c, n - 1 - r)):
        v = [f(r, c) for (r, c) in base]
        out.add(tuple(v)); out.add(tuple(reversed(v)))
    return out


_SNAKE_CACHE = {}


def snake_images(n):
    if n not in _SNAKE_CACHE:
        _SNAKE_CACHE[n] = _snake_images(n)
    return _SNAKE_CACHE[n]


# tort in [0,1] -> Metropolis inverse temperature on the turn count.
def _beta(tort):
    return 4.0 * (tort - 0.5)          # tort .25 -> -1.0 (straighter), .85 -> +1.4 (wigglier)


def backbite_hamiltonian(n, rng, tort=0.5, k=None, avoid_snake=True):
    """Hamiltonian path on the n x n grid by k backbite moves from the snake.

    Never fails, never falls back.  Returns a list of n*n (r, c) cells.
    """
    m = n * n
    if k is None:
        k = 6 * m
    beta = _beta(tort)
    path = _snake(n)
    pos = {cell: i for i, cell in enumerate(path)}
    t = _turns(path)

    for _ in range(k):
        tail = rng.random() < 0.5
        end = path[-1] if tail else path[0]
        cand = list(_nbrs(end[0], end[1], n))
        q = cand[rng.randrange(len(cand))]
        j = pos[q]
        if tail:
            if j == m - 2:                       # already the path neighbour: self-loop
                continue
            lo, hi = j + 1, m                    # reverse path[lo:hi]
            new_end = path[lo]
        else:
            if j == 1:
                continue
            lo, hi = 0, j
            new_end = path[hi - 1]
        # Full turn recount: only O(n*n) per move at these sizes, and provably
        # right, unlike windowed bookkeeping around the reversal joint.
        if tail:
            newpath = path[:lo] + path[lo:hi][::-1]
        else:
            newpath = path[lo:hi][::-1] + path[hi:]
        t_new = _turns(newpath)
        # MH: pi(x) ~ exp(beta * turns); proposal q(x->y) = 1/2 * 1/deg(end)
        log_a = beta * (t_new - t) + math.log(_degree(end, n)) - math.log(_degree(new_end, n))
        if log_a >= 0 or rng.random() < math.exp(log_a):
            path = newpath
            pos = {cell: i for i, cell in enumerate(path)}
            t = t_new

    if avoid_snake and tuple(path) in snake_images(n):
        return backbite_hamiltonian(n, rng, tort, k=k * 2, avoid_snake=False)
    return path


# --------------------------------------------------------------------- pipes
def cut_into_segments(path, c, rng):
    """Cut a Hamiltonian path into C contiguous segments with varied lengths.

    Every segment has length >= 2 (a real pipe needs distinct endpoints).
    Lengths sum to len(path) == N*N by construction.
    """
    total = len(path)
    assert total >= 2 * c, "grid too small for that many colors"
    mn = 3                       # minimum pipe length (cells)
    assert total >= mn * c, "grid too small for that many colors"
    # Start everyone near the mean, then redistribute a bounded amount so lengths
    # stay varied but realistic (avoid one giant pipe + tiny stubs). Each part
    # ends up in [mn, ~2*mean].
    base = total // c
    sizes = [base] * c
    sizes[-1] += total - sum(sizes)        # absorb remainder
    # random pairwise transfers keep the sum invariant
    for _ in range(c * 6):
        a, b = rng.randrange(c), rng.randrange(c)
        if a == b:
            continue
        amt = rng.randint(1, max(1, base // 2))
        if sizes[a] - amt >= mn and sizes[b] + amt <= 2 * base + base // 2:
            sizes[a] -= amt
            sizes[b] += amt
    assert all(s >= mn for s in sizes) and sum(sizes) == total
    segments = []
    i = 0
    for s in sizes:
        segments.append(path[i:i + s])
        i += s
    return segments
