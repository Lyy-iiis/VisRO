"""Independent answer and construction checks of the puzzle builders."""

import re
import numpy as np
from shared.puzzles.drawing import CODES, MAX_SIZE, image_size


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def independent_jigsaw(m, limit=2):
    holes = [tuple(p) for p in m['holes']]
    board = m['board']
    pool = m['pool']
    n = len(board)
    p = m['ports']
    choices = []
    for idx, edges in enumerate(pool):
        a = np.zeros((p + 2, p + 2), dtype=int)
        a[0, 1:-1] = edges[0]
        a[1:-1, -1] = edges[1]
        a[-1, 1:-1] = edges[2][::-1]
        a[1:-1, 0] = edges[3][::-1]
        for k in range(4):
            b = np.rot90(a, -k)
            edge = (tuple(b[0, 1:-1]), tuple(b[1:-1, -1]), tuple(b[-1, 1:-1][::-1]), tuple(b[1:-1, 0][::-1]))
            choices.append((idx, k, edge))
    known = {(r, c): board[r][c] for r in range(n) for c in range(n) if (r, c) not in holes}
    assignments = {}
    results = []

    def visit(unused):
        if len(results) >= limit:
            return
        if len(assignments) == len(holes):
            results.append(','.join('ABCDEFGHIJ'[assignments[h][0]] + str(assignments[h][1]) for h in holes))
            return
        candidates = []
        for r, c in holes:
            if (r, c) in assignments:
                continue
            options = []
            for idx, k, e in choices:
                if idx not in unused:
                    continue
                good = True
                for side, neighbor in enumerate(((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))):
                    other = assignments[neighbor][2] if neighbor in assignments else known.get(neighbor)
                    if other is not None and tuple(e[side]) != tuple(other[(side + 2) % 4][::-1]):
                        good = False
                        break
                if good:
                    options.append((idx, k, e))
            if not options:
                return
            candidates.append((len(options), (r, c), options))
        _, cell, options = min(candidates, key=lambda x: x[0])
        for item in options:
            assignments[cell] = item
            visit(unused - {item[0]})
            del assignments[cell]

    visit(set(range(len(pool))))
    return results


def check_paper(rec):
    # Independent inverse oracle: expand each punch through just the folds that
    # preceded it, clipping each preimage to the packet before that fold.
    m = rec['meta']
    n = m['n']
    box = [0, 0, 2 * n, 2 * n]
    folds = m['folds']
    states = []

    def inside(p, b):
        return b[0] < p[0] < b[2] and b[1] < p[1] < b[3]

    def reflection(p, step):
        x, y = p
        l = step['line']
        a = step['axis']
        if a == 'v':
            return 2 * l - x, y
        if a == 'h':
            return x, 2 * l - y
        if a == 'd':
            return y + l, x - l
        return l - y, l - x

    for i, step in enumerate(folds):
        require(step['before'] == box, 'Inconsistent packet bounds')
        before = box.copy()
        a = step['axis']
        l = step['line']
        k = step['keep']
        if a in ('v', 'h'):
            dim = 0 if a == 'v' else 1
            require(k in ('low', 'high'), 'Invalid fold direction')
            require(l == (box[dim] + box[dim + 2]) / 2, 'Crease is not current packet midpoint')
            box[dim + 2 if k == 'low' else dim] = l
        else:
            require(
                a in ('d', 'a') and i == len(folds) - 1 and box[2] - box[0] == box[3] - box[1], 'Invalid diagonal fold'
            )
            require(l == (box[0] - box[1] if a == 'd' else box[0] + box[3]) and k == 'lower', 'Wrong diagonal crease')
        require(step['after'] == box, 'Incorrect retained packet')
        states.append((before, box.copy()))
    holes = set()
    prior_stage = 0
    effects = []
    for event in m['events']:
        stage = event['stage']
        require(prior_stage < stage <= len(folds), 'Punch stages out of order')
        prior_stage = stage
        require(len({tuple(p) for p in event['centers']}) == len(event['centers']), 'Duplicate punches')
        new = set()
        for center in event['centers']:
            require(all(type(v) is int and v % 2 == 1 for v in center), 'Punch not at a cell center')
            require(inside(center, states[stage - 1][1]), 'Punch outside packet')
            a = folds[stage - 1]['axis']
            l = folds[stage - 1]['line']
            x, y = center
            require(a != 'd' or x - y < l, 'Punch outside/bordering diagonal packet')
            require(a != 'a' or x + y > l, 'Punch outside/bordering antidiagonal packet')
            points = {tuple(center)}
            for idx in range(stage - 1, -1, -1):
                expanded = set()
                for p in points:
                    for q in (p, reflection(p, folds[idx])):
                        if inside(q, states[idx][0]):
                            expanded.add(q)
                points = expanded
            for x, y in points:
                require(x % 2 == 1 and y % 2 == 1 and 0 < x < 2 * n and 0 < y < 2 * n, 'Invalid unfolded punch')
                new.add((y // 2, x // 2))
        require(bool(new - holes), 'A punch stage removes no new material')
        holes |= new
        effects.append(new)
    exclusive = []
    for i, effect in enumerate(effects):
        rest = set().union(*(other for j, other in enumerate(effects) if j != i))
        unique = effect - rest
        require(bool(unique), 'A punch stage is subsumed by later/other punches')
        exclusive.append(unique)
    if m['queries']:
        selected = set(map(tuple, m['queries']))
        require(all(bool(cells & selected) for cells in exclusive), 'Queries do not cover every punch stage')
    truth = [[int((r, c) in holes) for c in range(n)] for r in range(n)]
    require(truth == m['truth'], 'Forward material tracking disagrees with reverse unfold oracle')
    if m['options']:
        require(len(m['options']) == 4 and not m['queries'], 'Invalid choice configuration')
        require(len({str(o) for o in m['options']}) == 4, 'Duplicate unfolded option')
        require(all(sum(map(sum, o)) == len(holes) for o in m['options']), 'Hole-count shortcut')
        matches = [o == truth for o in m['options']]
        require(sum(matches) == 1, 'Unfolded answer is not unique')
        answer = 'ABCD'[matches.index(True)]
    else:
        require(len({tuple(p) for p in m['queries']}) == len(m['queries']), 'Duplicate queried cell')
        require(all(0 <= r < n and 0 <= c < n for r, c in m['queries']), 'Query outside original sheet')
        answer = ''.join(str(truth[r][c]) for r, c in m['queries'])
    require(rec['answer'] == answer, 'Paper folding answer mismatch')
    return dict(
        folds=len(folds),
        punch_stages=len(m['events']),
        holes=len(holes),
        queried_cells=len(m['queries']),
        minimum_stage_exclusive_holes=min(map(len, exclusive)),
    )


def check(rec):
    m = rec['meta']
    family = rec['builder_family']
    answer = rec['answer']
    if family == 'staged_paper_punches':
        return check_paper(rec)
    if family == 'partial_reflections':
        truth = []
        sources = []
        for rr, cc in m['queries']:
            r, c = rr, cc
            hits = 0
            for axis, r0, r1, c0, c1 in reversed(m['operations']):
                if r0 <= r < r1 and c0 <= c < c1:
                    hits += 1
                    if axis == 0:
                        r = r0 + r1 - 1 - r
                    else:
                        c = c0 + c1 - 1 - c
            require(hits >= m['minimum_query_operations'], 'Insufficient query trajectory depth')
            truth.append(CODES[m['grid'][r][c]])
            sources.append((r, c))
        require(''.join(truth) == answer, 'Reverse reflection trace mismatch')
        require(len(set(sources)) == len(sources), 'Duplicate query source')
        return dict(steps=len(m['operations']), queries=len(sources))
    if family == 'composed_boolean':

        def apply(a, b, k):
            table = ((0, 1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0))[k]
            return [[table[2 * x + y] for x, y in zip(ar, br)] for ar, br in zip(a, b)]

        inferred = []
        for i in range(len(m['operators'])):
            demos = m['demonstrations'][2 * i : 2 * i + 2]
            matches = [k for k in range(4) if all(apply(a, b, k) == c for a, b, c in demos)]
            require(len(matches) == 1, 'Operator not identified')
            inferred.append(matches[0])
        truth = m['inputs'][0]
        n = m['n']
        for i, k in enumerate(inferred):
            for _ in range(m['rotations'][i]):
                truth = [[truth[n - 1 - c][r] for c in range(n)] for r in range(n)]
            truth = apply(truth, m['inputs'][i + 1], k)
        require(truth == m['truth'], 'Boolean composition mismatch')
        if m['options']:
            hits = [x == truth for x in m['options']]
            require(sum(hits) == 1, 'Options are not unique')
            expected = 'ABCD'[hits.index(True)]
            arrays = list(map(np.asarray, m['options']))
            scores = [sum(np.count_nonzero(a != b) for b in arrays) for a in arrays]
            require(len(set(scores)) == 1, 'Option medoid shortcut')
        else:
            expected = '/'.join(''.join(map(str, r)) for r in truth)
        require(answer == expected, 'Matrix answer mismatch')
        return dict(stages=len(inferred), cells=n * n)
    if family == 'joint_edge_jigsaw':
        found = independent_jigsaw(m)
        require(found == [answer], 'Joint jigsaw is not uniquely solved by the gold answer')
        return dict(holes=len(m['holes']), candidates=len(m['pool']))
    raise AssertionError(f'Unknown family: {family}')


def check_render(rec, im):
    expected = image_size(rec['subtype'], rec['difficulty'])
    require(im.mode == 'RGB' and im.size == (expected, expected) and max(im.size) <= MAX_SIZE, 'Image profile mismatch')
    require(
        all(re.fullmatch(r'(?:[A-Z]{0,2}[0-9]{0,2}|[=?])', text) for text in rec['image_labels']),
        'Explanatory text remains in the question image',
    )
    metrics = dict(width=im.width, height=im.height)
    if rec['builder_family'] == 'staged_paper_punches':
        m = rec['meta']
        panels = m['render']['packet_panels']
        punched = [p for p in panels if p['has_punches']]
        require(
            all(p['min_cell_pixels'] >= 12 and p['punch_radius_pixels'] >= 2.5 for p in punched),
            'Punch diagram too small',
        )
        metrics['min_punched_cell_pixels'] = min(p['min_cell_pixels'] for p in punched)
        require(len(m['options']) == len(m['render']['option_boxes']), 'Missing option image')
        for option, box in zip(m['options'], m['render']['option_boxes']):
            x0, y0, x1, y1 = box
            require(0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height, 'Option outside canvas')
            side = x1 - x0 - 1
            cell = side / m['n']
            for r, row in enumerate(option):
                for c, hole in enumerate(row):
                    pixel = im.getpixel((int(x0 + (c + 0.5) * cell), int(y0 + (r + 0.5) * cell)))
                    require((min(pixel) > 245) == bool(hole), 'Rendered hole differs from option geometry')
    return metrics
