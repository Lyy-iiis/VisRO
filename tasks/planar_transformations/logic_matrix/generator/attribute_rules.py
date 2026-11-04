"""The attribute / overlay rule texts of the low tiers and their exhaustive rule-family checks."""

import itertools, math

ATTR_RULES = (
    'Use ONLY the following rule family. Exactly {changing} of the five attributes (pip count, color, outline shape, size, rotation) change; every other attribute is constant throughout the matrix. '
    'For a changing pip count, moving one column right adds one pip and moving one row down adds one pip. '
    'A changing color or outline shape repeats a fixed cycle of THREE DISTINCT values from left to right; every next row starts one or two positions later in that same cycle, using the same row shift throughout. '
    'A changing size repeats small, medium, large in that order from left to right; every next row starts one or two positions later, using the same row shift throughout. '
    'A changing rotation adds a fixed 30, 45, 60, or 90 degrees clockwise per column; every next row starts 40 degrees clockwise from the previous row start. Rotation moves the outline and its pip ring together. '
    'Complete the missing bottom-right cell. Return the matching option letter A, B, C, or D.'
)
OVERLAY_RULES = (
    'Filled tiles represent 1 and empty tiles 0. In BOTH rows, the third figure is obtained from the first two by the SAME pixelwise operation. '
    'The ONLY allowed operations are OR (filled in either input), AND (filled in both), XOR (filled in exactly one), and left-minus-right (filled on the left and empty on the right). '
    'Infer the operation from the complete first row and fill the missing third figure of the second row. Return A, B, C, or D.'
)


def logic_question(tier, family):
    return ATTR_RULES.format(changing=3 if tier == 'easy' else 4) if family == 'explicit_attribute' else OVERLAY_RULES


def require(value, message):
    if not value:
        raise AssertionError(message)


def period(glyph):
    count = int(glyph['count'])
    require(count > 0, 'Pip count must be positive')
    if glyph['shape'] == 'circle':
        return 360 / count
    symmetry = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 'star': 5}[glyph['shape']]
    return 360 / math.gcd(symmetry, count)


def same_rotation(a, b, glyph):
    p = period(glyph)
    d = (a - b) % p
    return min(d, p - d) < 1e-6


def glyph_key(glyph):
    return (
        glyph['shape'],
        tuple(glyph['color']),
        int(glyph['count']),
        round(float(glyph['size']), 6),
        round(float(glyph['rot']) % period(glyph), 6),
    )


def close(a, b):
    return abs(float(a) - float(b)) < 1e-6


def attribute_solutions(meta):
    grid = meta['grid_attr']
    nr, nc = meta['rows'], meta['cols']
    require(nc == 3 and nr in (2, 3), 'Unsupported matrix shape')
    visible = [(r, c) for r in range(nr) for c in range(nc) if (r, c) != (nr - 1, nc - 1)]
    require(len(grid) == nr and all(len(row) == nc for row in grid), 'Bad attribute grid')
    keys = ('count', 'color', 'shape', 'size', 'rot')
    candidates = {}
    for name in keys:
        programs = []
        initial = grid[0][0][name]
        if name == 'rot':
            p = period(grid[0][0])
            bases = [(initial + k * p) % 360 for k in range(round(360 / p))]
            for base in bases:
                programs.append((False, lambda r, c, b=base: b))
                for step in (30, 45, 60, 90):
                    programs.append((True, lambda r, c, b=base, s=step: (b + 40 * r + s * c) % 360))
        else:
            programs.append((False, lambda r, c, x=initial: x))
            if name == 'count':
                programs.append((True, lambda r, c, x=initial: x + r + c))
            elif name in ('color', 'shape'):
                cycle = [grid[0][c][name] for c in range(3)]
                if len({str(x) for x in cycle}) == 3:
                    for shift in (1, 2):
                        programs.append((True, lambda r, c, v=cycle, s=shift: v[(c + s * r) % 3]))
            else:
                cycle = (0.85, 1.0, 1.15)
                for start in range(3):
                    for shift in (1, 2):
                        programs.append((True, lambda r, c, v=cycle, s=shift, k=start: v[(k + c + s * r) % 3]))
        accepted = []
        for changing, fn in programs:
            good = True
            for r, c in visible:
                predicted, observed = fn(r, c), grid[r][c][name]
                same = (
                    same_rotation(predicted, observed, grid[r][c])
                    if name == 'rot'
                    else close(predicted, observed) if name == 'size' else predicted == observed
                )
                if not same:
                    good = False
                    break
            if good:
                accepted.append((changing, fn(nr - 1, nc - 1)))
        require(bool(accepted), 'No allowed program for attribute ' + name)
        candidates[name] = accepted
    answers = {}
    valid_programs = 0
    for combination in itertools.product(*(candidates[k] for k in keys)):
        if sum(changing for changing, _ in combination) != meta['changing_attributes']:
            continue
        glyph = {name: value for name, (_, value) in zip(keys, combination)}
        answers[glyph_key(glyph)] = glyph
        valid_programs += 1
    require(valid_programs > 0, 'No program obeys the changing-attribute count')
    return answers, valid_programs


def check_logic_low(rec):
    m = rec['meta']
    family = m['family']
    tier = rec['difficulty']
    require(tier in ('easy', 'medium'), 'Wrong logic tier')
    require(rec['question'] == logic_question(tier, family), 'The public finite rule contract changed')
    if family == 'explicit_attribute':
        require(
            m['rows'] == (2 if tier == 'easy' else 3) and m['changing_attributes'] == (3 if tier == 'easy' else 4),
            'Changed attribute difficulty',
        )
        answers, programs = attribute_solutions(m)
        require(len(answers) == 1, 'Allowed programs predict different missing glyphs')
        require(glyph_key(m['grid_attr'][-1][-1]) in answers, 'Planted glyph does not follow public rules')
        hits = [glyph_key(o['attr']) in answers for o in m['options']]
    elif family == 'explicit_overlay':
        rows = m['row_masks']
        dims = {
            'square': (4, 4) if tier == 'easy' else (5, 5),
            'triangle': (4, 6) if tier == 'easy' else (5, 8),
            'hexagon': (3, 4) if tier == 'easy' else (4, 5),
        }
        require(m['grid_kind'] in dims and (m['R'], m['C']) == dims[m['grid_kind']], 'Changed overlay dimensions')
        require(len(rows) == 2 and all(len(row) == 3 for row in rows), 'Bad overlay rows')
        a, b, c = rows[0]
        n = m['R'] * m['C']
        require(all(len(g) == n and set(g) <= {0, 1} for row in rows for g in row), 'Malformed overlay mask')
        tables = ((0, 1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0))
        apply = lambda left, right, tab: [tab[2 * x + y] for x, y in zip(left, right)]
        allowed = [tab for tab in tables if apply(a, b, tab) == c]
        require(len(allowed) == 1, 'Overlay demonstration is ambiguous')
        gold = apply(rows[1][0], rows[1][1], allowed[0])
        require(gold == rows[1][2], 'Overlay gold mismatch')
        hits = [o['mask'] == gold for o in m['options']]
        programs = len(allowed)
    else:
        raise AssertionError('Unknown explicit logic family')
    require(len(hits) == 4 and sum(hits) == 1, 'No unique matching answer option')
    require(rec['answer'] == 'ABCD'[hits.index(True)], 'Wrong logic answer letter')
    return dict(valid_programs=programs, unique_answer=True)
