"""Reflection and matrix families; answers computed before rendering."""

import numpy as np
from shared.puzzles.drawing import CODES, canvas, draw_grid, label, image_size, record


def reflect_grid(a, op):
    axis, r0, r1, c0, c1 = op
    out = a.copy()
    out[r0:r1, c0:c1] = np.flip(a[r0:r1, c0:c1], axis=axis)
    return out


def mirrors(tier, rng):
    rank = ('hard', 'xhard', 'xxhard').index(tier)
    n = (8, 10, 12)[rank]
    steps = (2, 4, 6)[rank]
    nq = (4, 8, 12)[rank]
    grid = np.array([[rng.randrange(1, 6) if rng.random() < 0.88 else 0 for _ in range(n)] for _ in range(n)])
    identity = np.arange(n * n).reshape(n, n)
    minimum_hits = (2, 3, 4)[rank]
    for attempt in range(500):
        ops = []
        truth = grid.copy()
        mapping = identity.copy()
        for k in range(steps):
            r0 = rng.randrange(0, n // 2)
            r1 = rng.randrange(n // 2 + 1, n + 1)
            c0 = rng.randrange(0, n // 2)
            c1 = rng.randrange(n // 2 + 1, n + 1)
            op = [k % 2, r0, r1, c0, c1]
            ops.append(op)
            truth = reflect_grid(truth, op)
            mapping = reflect_grid(mapping, op)
        eligible = []
        for rr in range(n):
            for cc in range(n):
                r, c = rr, cc
                hits = 0
                for axis, r0, r1, c0, c1 in reversed(ops):
                    if r0 <= r < r1 and c0 <= c < c1:
                        hits += 1
                        if axis == 0:
                            r = r0 + r1 - 1 - r
                        else:
                            c = c0 + c1 - 1 - c
                if hits >= minimum_hits and (r, c) != (rr, cc):
                    eligible.append((rr, cc))
        symmetries = [np.rot90(identity, k) for k in range(4)] + [np.fliplr(np.rot90(identity, k)) for k in range(4)]
        if len(eligible) >= nq and all(not np.array_equal(mapping, g) for g in symmetries):
            break
    else:
        raise RuntimeError('Cannot sample sufficiently deep reflection trajectories')
    queries = rng.sample(eligible, nq)
    answer = ''.join(CODES[truth[r, c]] for r, c in queries)
    im = canvas()
    cell = 352 // n
    draw_grid(im, grid, (im.width - n * cell) // 2, 48, cell, coords=True)
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
    q = (
        'The image shows the INITIAL grid. Coordinates are 1-based from the top-left. Apply these operations in order; each modifies only its stated rectangle and uses the current grid:\n'
        + '\n'.join(desc)
        + '\nAfter all operations, read these cells in order: '
        + ', '.join(f'({r+1},{c+1})' for r, c in queries)
        + '. Return one string of '
        + str(nq)
        + ' color codes: R=red, B=blue, G=green, Y=yellow, P=purple, and .=white.'
    )
    rec = record(
        'mirroring_patterns',
        tier,
        q,
        answer,
        dict(
            family='partial_reflections',
            grid=grid.tolist(),
            operations=ops,
            queries=queries,
            final_grid=truth.tolist(),
            source_map=mapping.tolist(),
            minimum_query_operations=minimum_hits,
        ),
    )
    return rec, im


def boolean(a, b, op):
    return (a | b, a & b, a ^ b, a & (1 - b))[op]


def mask(rng, n):
    return np.array([[rng.randrange(2) for _ in range(n)] for _ in range(n)], dtype=np.uint8)


def logic(tier, rng):
    rank = ('hard', 'xhard', 'xxhard').index(tier)
    n = (5, 6, 7)[rank]
    stages = rank + 1
    operators = rng.sample(range(4), stages)
    demos = []
    for op in operators:
        for _ in range(2):
            for attempt in range(100):
                a, b = mask(rng, n), mask(rng, n)
                values = [boolean(a, b, k) for k in range(4)]
                if len({x.tobytes() for x in values}) == 4:
                    break
            else:
                raise RuntimeError('Ambiguous operator demonstration')
            demos.append([a.tolist(), b.tolist(), values[op].tolist()])
    inputs = [mask(rng, n) for _ in range(stages + 1)]
    rotations = [0] + [rng.choice((1, 2, 3)) for _ in range(stages - 1)]
    truth = inputs[0]
    for k, op in enumerate(operators):
        truth = boolean(np.rot90(truth, -rotations[k]), inputs[k + 1], op)
    options = []
    if rank < 2:
        # Equal pairwise distances: option medoid cannot identify the truth.
        pos = rng.sample(range(n * n), 4)
        a = truth.copy().reshape(-1)
        b = a.copy()
        c = a.copy()
        d = a.copy()
        for i in pos[:2]:
            b[i] ^= 1
            d[i] ^= 1
        for i in pos[2:]:
            c[i] ^= 1
            d[i] ^= 1
        options = [x.reshape(n, n) for x in (a, b, c, d)]
        rng.shuffle(options)
        answer = 'ABCD'[next(i for i, x in enumerate(options) if np.array_equal(x, truth))]
    else:
        answer = '/'.join(''.join(map(str, row)) for row in truth.tolist())
    im = canvas(image_size('logic_matrix', tier))
    large = im.width > 448
    cell = (12, 12, 11)[rank]
    tile = n * cell
    gap = 18 if large else 14
    rowh = tile + (42 if large else 32)
    for j, (a, b, c) in enumerate(demos):
        y = 24 + (j // 2) * rowh
        opname = 'XYZ'[j // 2]
        left = 10 + (j % 2) * (im.width // 2)
        for k, g in enumerate((a, b, c)):
            draw_grid(im, g, left + k * (tile + gap), y, cell, binary=True)
        label(im, opname, (left + tile + 3, y + tile / 2 - 8), 12)
        label(im, '=', (left + 2 * tile + gap + 3, y + tile / 2 - 8), 12)
    top = 24 + stages * rowh + 36
    stride = im.width // 4
    for k, g in enumerate(inputs):
        x = k * stride + (stride - tile) // 2
        draw_grid(im, g, x, top, cell, binary=True)
        label(im, 'I' + str(k + 1), (x + tile / 2 - 8, top - 20), 12)
    if options:
        opttop = top + tile + 58
        for k, g in enumerate(options):
            x = k * stride + (stride - tile) // 2
            label(im, 'ABCD'[k], (x + tile / 2 - 4, opttop - 20), 12)
            draw_grid(im, g, x, opttop, cell, binary=True)
        if opttop + tile > im.height - 8:
            raise RuntimeError('Matrix rendering overflow')
    q = 'Each operator X, Y or Z used in the image is fixed across its two demonstrations. Every operator is one of pixelwise OR, AND, XOR, or left-minus-right (a cell is filled iff it is filled on the left and empty on the right). Infer each operator from its demonstrations. The input tiles below the demonstrations are labeled I1, I2, I3, I4 as needed; any answer options form the bottom row labeled A–D. Start with input I1. '
    for k, turn in enumerate(rotations):
        q += f'Step {k+1}: rotate the current tile {turn*90} degrees clockwise, then apply operator {"XYZ"[k]} with input I{k+2} as the right operand. '
    q += (
        'Which bottom option is the resulting tile? Return A, B, C, or D.'
        if options
        else f'Reconstruct the final {n} by {n} tile. Return {n} rows of {n} binary digits, separated by /; 1 means filled and 0 empty, with rows top-to-bottom and columns left-to-right.'
    )
    rec = record(
        'logic_matrix',
        tier,
        q,
        answer,
        dict(
            family='composed_boolean',
            n=n,
            operators=operators,
            demonstrations=demos,
            inputs=[x.tolist() for x in inputs],
            rotations=rotations,
            truth=truth.tolist(),
            options=[x.tolist() for x in options],
        ),
    )
    return rec, im
