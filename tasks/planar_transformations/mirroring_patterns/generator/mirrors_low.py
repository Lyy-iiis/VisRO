"""Easy and medium tiers of the partial-reflection family.

shared/puzzles/symbolic.py builds `partial_reflections` for hard/xhard/xxhard.
This module builds the two low tiers with the SAME question, the same metadata contract and the same colour-code answer, only
with relaxed parameters, so `oracles.check` (family `partial_reflections`) and
`adapter.core_key` both apply unchanged.
"""

import numpy as np
from shared.puzzles import drawing as draw
from shared.puzzles.runtime import record, rng_for, plain

# Every dimension stays at or below the hard tier (n=8, steps=2, nq=4, hits=2).
CONFIG = {'easy': dict(n=4, steps=1, nq=3, minimum_hits=1), 'medium': dict(n=6, steps=2, nq=3, minimum_hits=1)}


def reflect(a, op):
    axis, r0, r1, c0, c1 = op
    out = a.copy()
    out[r0:r1, c0:c1] = np.flip(out[r0:r1, c0:c1], axis=axis)
    return out


def trace(ops, rr, cc):
    # Walk the query position backwards to the initial cell it reads from.
    r, c, hits = rr, cc, 0
    for axis, r0, r1, c0, c1 in reversed(ops):
        if r0 <= r < r1 and c0 <= c < c1:
            hits += 1
            if axis == 0:
                r = r0 + r1 - 1 - r
            else:
                c = c0 + c1 - 1 - c
    return r, c, hits


def question(meta):
    ops = meta['operations']
    nq = len(meta['queries'])
    single = len(ops) == 1
    desc = []
    for k, (axis, r0, r1, c0, c1) in enumerate(ops, 1):
        desc.append(
            f'{k}. Within rows {r0+1}–{r1} and columns {c0+1}–{c1}, '
            + (
                'reverse the order of rows (top-bottom reflection)'
                if axis == 0
                else 'reverse the order of columns (left-right reflection)'
            )
            + '.'
        )
    head = 'Apply this operation; it modifies' if single else 'Apply these operations in order; each modifies'
    tail = 'After the operation, read' if single else 'After all operations, read'
    return (
        'The image shows the INITIAL grid. Coordinates are 1-based from the top-left. '
        + head
        + ' only its stated rectangle and uses the current grid:\n'
        + '\n'.join(desc)
        + '\n'
        + tail
        + ' these cells in order: '
        + ', '.join(f'({r+1},{c+1})' for r, c in meta['queries'])
        + '. Return one string of '
        + str(nq)
        + ' color codes: R=red, B=blue, G=green, Y=yellow, P=purple, and .=white.'
    )


def build(tier, seed, index):
    cfg = CONFIG[tier]
    n, steps, nq, hits = cfg['n'], cfg['steps'], cfg['nq'], cfg['minimum_hits']
    rng = rng_for('mirrors-low', tier, seed, index)
    identity = np.arange(n * n).reshape(n, n)
    symmetries = [np.rot90(identity, k) for k in range(4)] + [np.fliplr(np.rot90(identity, k)) for k in range(4)]
    for attempt in range(500):
        grid = np.array([[rng.randrange(1, 6) if rng.random() < 0.88 else 0 for _ in range(n)] for _ in range(n)])
        # A single reflection would otherwise always be the top-bottom one.
        parity = rng.randrange(2)
        ops = []
        truth = grid.copy()
        mapping = identity.copy()
        for k in range(steps):
            r0 = rng.randrange(0, n // 2)
            r1 = rng.randrange(n // 2 + 1, n + 1)
            c0 = rng.randrange(0, n // 2)
            c1 = rng.randrange(n // 2 + 1, n + 1)
            op = [(k + parity) % 2, r0, r1, c0, c1]
            ops.append(op)
            truth = reflect(truth, op)
            mapping = reflect(mapping, op)
        if any(np.array_equal(mapping, g) for g in symmetries):
            continue
        eligible = []
        for rr in range(n):
            for cc in range(n):
                r, c, seen = trace(ops, rr, cc)
                if seen >= hits and (r, c) != (rr, cc):
                    eligible.append((rr, cc))
        if len(eligible) < nq:
            continue
        # Reading order is canonical, so one core has exactly one gold string.
        queries = sorted(rng.sample(eligible, nq))
        answer = ''.join(draw.CODES[truth[r, c]] for r, c in queries)
        # Reject items that "reading the initial grid" would already answer.
        if answer == ''.join(draw.CODES[grid[r, c]] for r, c in queries):
            continue
        m = dict(
            family='partial_reflections',
            grid=plain(grid),
            operations=ops,
            queries=[list(q) for q in queries],
            final_grid=plain(truth),
            source_map=plain(mapping),
            minimum_query_operations=hits,
            requirements=cfg,
            attempts=attempt + 1,
        )
        rec = record('mirroring_patterns', tier, seed, index, question(m), answer, m, 'exact')
        rec['builder_family'] = 'partial_reflections'
        return rec
    raise RuntimeError('Cannot sample sufficiently deep low-tier reflection trajectories')


def panel(grid):
    n = len(grid)
    cell = 352 // n
    im = draw.canvas()
    draw.draw_grid(im, grid, (im.width - n * cell) // 2, 48, cell, coords=True)
    return im
