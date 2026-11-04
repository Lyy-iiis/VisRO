"""Independent truth-table solver and exhaustive rule/answer audit for the boolean grids."""

import itertools, math, re
from PIL import Image

TABLES = ((0, 1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0))


def operation(a, b, op):
    return [[TABLES[op][2 * x + y] for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def rotate(a, turns):
    out = [list(row) for row in a]
    for _ in range(turns):
        out = [list(row) for row in zip(*out[::-1])]
    return out


def solve(inputs, ops, rotations, skip=None):
    out = inputs[0]
    for i, op in enumerate(ops):
        out = rotate(out, rotations[i])
        if i != skip:
            out = operation(out, inputs[i + 1], op)
    return out


def distance(a, b):
    return sum(x != y for ar, br in zip(a, b) for x, y in zip(ar, br))


def answer(grid):
    return '/'.join(''.join(map(str, row)) for row in grid)


def score(rec, text):
    clean = re.sub(r'\s+', '', str(text))
    n = rec['meta']['n']
    parsed = bool(re.fullmatch(r'[01]{' + str(n) + r'}(?:/[01]{' + str(n) + r'}){' + str(n - 1) + r'}', clean))
    return dict(
        correct=parsed and clean == rec['answer'],
        parsed=parsed,
        component_correct=(
            sum(x == y for x, y in zip(clean.replace('/', ''), rec['answer'].replace('/', ''))) if parsed else 0
        ),
        component_total=n * n,
    )


def check(rec, image_path=None):
    m = rec['meta']
    n = m['n']
    cfg = m['requirements']
    stages = len(m['operators'])
    assert rec['difficulty'] in ('hard', 'xhard')
    assert len(m['inputs']) == stages + 1 and len(m['demonstrations']) == stages and len(m['rotations']) == stages
    grids = (
        list(m['inputs']) + [m['truth']] + [grid for demos in m['demonstrations'] for panel in demos for grid in panel]
    )
    for grid in grids:
        assert len(grid) == n and all(
            len(row) == n and all(type(v) is int and v in (0, 1) for v in row) for row in grid
        )
    assert n == (4 if rec['difficulty'] == 'hard' else 5)
    expected_cfg = (
        dict(
            n=4,
            stages=2,
            size=448,
            min_input_dependency=6,
            min_stage_effect=4,
            min_wrong_rule_effect=2,
            min_rotation_effect=2,
            min_distinct_operators=2,
        )
        if rec['difficulty'] == 'hard'
        else dict(
            n=5,
            stages=3,
            size=588,
            min_input_dependency=9,
            min_stage_effect=5,
            min_wrong_rule_effect=3,
            min_rotation_effect=3,
            min_distinct_operators=2,
        )
    )
    assert cfg == expected_cfg, 'Difficulty constraints were changed'
    assert len(set(m['operators'])) >= cfg['min_distinct_operators']
    assert stages == (2 if rec['difficulty'] == 'hard' else 3)
    assert m['rotations'][0] == 0 and all(k in (1, 2, 3) for k in m['rotations'][1:])
    for a in m['inputs']:
        assert len(a) == n and all(len(row) == n and all(type(v) is int and v in (0, 1) for v in row) for row in a)
    allowed = []
    single = []
    for demos in m['demonstrations']:
        possible = []
        for a, b, c in demos:
            matches = {op for op in range(4) if operation(a, b, op) == c}
            assert len(matches) >= 2, 'A single demonstration identifies the rule'
            possible.append(matches)
            single.append(len(matches))
        assert len(possible) == 2
        intersection = possible[0] & possible[1]
        assert len(intersection) == 1, 'Combined demonstrations are ambiguous'
        allowed.append(intersection)
    # Exhaust the complete declared hypothesis space, not just the stored rule.
    valid = []
    for ops in itertools.product(range(4), repeat=stages):
        if all(op in allowed[i] for i, op in enumerate(ops)):
            valid.append(ops)
    assert len(valid) == 1 and list(valid[0]) == m['operators']
    truth = solve(m['inputs'], valid[0], m['rotations'])
    assert truth == m['truth'] and answer(truth) == rec['answer']
    expected = dict(stage_effect=[], min_wrong_rule_effect=[], rotation_effect=[], input_dependency=[])
    for i, op in enumerate(valid[0]):
        effect = distance(truth, solve(m['inputs'], valid[0], m['rotations'], skip=i))
        assert effect >= cfg['min_stage_effect']
        expected['stage_effect'].append(effect)
        effects = []
        for wrong in range(4):
            if wrong != op:
                ops = list(valid[0])
                ops[i] = wrong
                effects.append(distance(truth, solve(m['inputs'], ops, m['rotations'])))
        assert min(effects) >= cfg['min_wrong_rule_effect']
        expected['min_wrong_rule_effect'].append(min(effects))
    for i, turn in enumerate(m['rotations']):
        if not turn:
            continue
        effects = []
        for wrong in range(4):
            if wrong != turn:
                turns = list(m['rotations'])
                turns[i] = wrong
                effects.append(distance(truth, solve(m['inputs'], valid[0], turns)))
        assert min(effects) >= cfg['min_rotation_effect']
        expected['rotation_effect'].append(min(effects))
    for i, a in enumerate(m['inputs']):
        hits = 0
        # Independent one-cell interventions: do not reuse generator's whole-
        # input complement shortcut for counting relevant output cells.
        for r in range(n):
            for c in range(n):
                inputs = [[row[:] for row in grid] for grid in m['inputs']]
                inputs[i][r][c] ^= 1
                changed = distance(truth, solve(inputs, valid[0], m['rotations']))
                assert changed in (0, 1)
                hits += changed
        assert hits >= cfg['min_input_dependency']
        expected['input_dependency'].append(hits)
    last = m['inputs'][-1]
    op = valid[0][-1]
    shortcut = sum(TABLES[op][v] == TABLES[op][2 + v] for row in last for v in row)
    assert shortcut <= math.floor(0.4 * n * n)
    expected['last_input_alone_determines'] = shortcut
    assert expected == m['significance']
    assert math.ceil(0.25 * n * n) <= sum(map(sum, truth)) <= math.floor(0.75 * n * n)
    assert score(rec, rec['answer'])['correct'] and not rec['options']
    if image_path:
        with Image.open(image_path) as image:
            assert image.mode == 'RGB' and image.size == (cfg['size'], cfg['size'])
            assert max(image.size) <= 588
            expected_grids = {f'input-{i}': a for i, a in enumerate(m['inputs'])}
            for i, demos in enumerate(m['demonstrations']):
                for j, panel in enumerate(demos):
                    for k, a in enumerate(panel):
                        expected_grids[f'demo-{i}-{j}-{k}'] = a
            assert len(rec['render_regions']) == len(expected_grids)
            assert {r['name'] for r in rec['render_regions']} == set(expected_grids)
            for a, b in itertools.combinations(rec['render_regions'], 2):
                ax, ay, ax1, ay1 = a['bounds']
                bx, by, bx1, by1 = b['bounds']
                assert min(ax1, bx1) <= max(ax, bx) or min(ay1, by1) <= max(ay, by), 'Rendered grids overlap'
            for region in rec['render_regions']:
                a = expected_grids[region['name']]
                assert a == region['values']
                x, y, x1, y1 = region['bounds']
                cell = region['cell']
                assert 0 <= x < x1 < image.width and 0 <= y < y1 < image.height
                for r, row in enumerate(a):
                    for c, v in enumerate(row):
                        assert image.getpixel((x + c * cell + cell // 2, y + r * cell + cell // 2)) == (
                            (40, 65, 100) if v else (249, 249, 246)
                        )
            assert set(rec['image_labels']) <= set('XYZ') | {'=', 'I1', 'I2', 'I3', 'I4'}
    return dict(
        hypotheses_enumerated=4**stages,
        valid_rule_combinations=1,
        unique_answer=True,
        single_demo_candidates=single,
        **expected,
    )
