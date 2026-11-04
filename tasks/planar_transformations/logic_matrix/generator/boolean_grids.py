"""Boolean-grid logic matrices (hard, xhard): small full grids and necessary composition."""

from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = _ROOT / 'assets/fonts/DejaVuSans.ttf'
# fixed RNG namespace: every item is seeded from it, so it must not change
_RNG_NAMESPACE = seed_text(563027601304329900117401072539891252)
LABEL = 'boolean_grids'
CONFIG = {
    'hard': dict(
        n=4,
        stages=2,
        size=448,
        min_input_dependency=6,
        min_stage_effect=4,
        min_wrong_rule_effect=2,
        min_rotation_effect=2,
        min_distinct_operators=2,
    ),
    'xhard': dict(
        n=5,
        stages=3,
        size=588,
        min_input_dependency=9,
        min_stage_effect=5,
        min_wrong_rule_effect=3,
        min_rotation_effect=3,
        min_distinct_operators=2,
    ),
}


def apply(a, b, op):
    if op == 0:
        return a | b
    if op == 1:
        return a & b
    if op == 2:
        return a ^ b
    return a & (1 - b)


def execute(inputs, ops, rotations, skip=None):
    cur = np.asarray(inputs[0], dtype=np.uint8).copy()
    for i, op in enumerate(ops):
        cur = np.rot90(cur, -rotations[i])
        if i != skip:
            cur = apply(cur, np.asarray(inputs[i + 1], dtype=np.uint8), op)
    return cur


def candidates(a, b, out):
    return [op for op in range(4) if np.array_equal(apply(a, b, op), out)]


def demonstrations(op, n, rng):
    # Each panel alone leaves exactly two Boolean operations possible.
    # Their intersection is the unique intended operation.
    domains = ([0, 1, 2] if op in (0, 2) else [0, 1], [0, 3])
    panels = []
    for domain in domains:
        flat = list(domain) + [rng.choice(domain) for _ in range(n * n - len(domain))]
        rng.shuffle(flat)
        types = np.asarray(flat, dtype=np.uint8).reshape(n, n)
        a, b = types // 2, types % 2
        out = apply(a, b, op)
        assert len(candidates(a, b, out)) == 2
        panels.append([a.tolist(), b.tolist(), out.tolist()])
    rng.shuffle(panels)
    assert set.intersection(*(set(candidates(np.array(a), np.array(b), np.array(c))) for a, b, c in panels)) == {op}
    return panels


def significance(inputs, ops, rotations):
    truth = execute(inputs, ops, rotations)
    count = lambda a: int(np.count_nonzero(a != truth))
    stage_effect = [count(execute(inputs, ops, rotations, skip=i)) for i in range(len(ops))]
    wrong_rules = []
    for i, op in enumerate(ops):
        effects = []
        for wrong in range(4):
            if wrong == op:
                continue
            altered = list(ops)
            altered[i] = wrong
            effects.append(count(execute(inputs, altered, rotations)))
        wrong_rules.append(min(effects))
    rotation_effect = []
    for i, rotation in enumerate(rotations):
        if not rotation:
            continue
        effects = []
        for wrong in range(4):
            if wrong == rotation:
                continue
            altered = list(rotations)
            altered[i] = wrong
            effects.append(count(execute(inputs, ops, altered)))
        rotation_effect.append(min(effects))
    # Pixelwise operations and rotations map each input cell to at most one
    # output cell. Complementing a whole input counts all local dependencies.
    dependency = []
    for i in range(len(inputs)):
        altered = [np.array(a, dtype=np.uint8) for a in inputs]
        altered[i] = 1 - altered[i]
        dependency.append(count(execute(altered, ops, rotations)))
    last = np.asarray(inputs[-1])
    z = np.zeros_like(last)
    o = np.ones_like(last)
    shortcut = int(np.count_nonzero(apply(z, last, ops[-1]) == apply(o, last, ops[-1])))
    return dict(
        stage_effect=stage_effect,
        min_wrong_rule_effect=wrong_rules,
        rotation_effect=rotation_effect,
        input_dependency=dependency,
        last_input_alone_determines=shortcut,
    )


def acceptable(stats, truth, cfg):
    cells = cfg['n'] ** 2
    filled = int(truth.sum())
    return (
        math.ceil(0.25 * cells) <= filled <= math.floor(0.75 * cells)
        and min(stats['input_dependency']) >= cfg['min_input_dependency']
        and min(stats['stage_effect']) >= cfg['min_stage_effect']
        and min(stats['min_wrong_rule_effect']) >= cfg['min_wrong_rule_effect']
        and min(stats['rotation_effect']) >= cfg['min_rotation_effect']
        and stats['last_input_alone_determines'] <= math.floor(0.4 * cells)
    )


def make(tier, seed, index):
    cfg = CONFIG[tier]
    n = cfg['n']
    rng = random.Random(f'{_RNG_NAMESPACE}/{tier}/{seed}/{index}')
    for attempt in range(10000):
        ops = [rng.randrange(4) for _ in range(cfg['stages'])]
        if len(set(ops)) < cfg['min_distinct_operators']:
            continue
        rotations = [0] + [rng.choice((1, 2, 3)) for _ in range(cfg['stages'] - 1)]
        inputs = []
        for _ in range(cfg['stages'] + 1):
            count = rng.randint(math.ceil(0.3 * n * n), math.floor(0.7 * n * n))
            a = np.zeros(n * n, dtype=np.uint8)
            a[rng.sample(range(n * n), count)] = 1
            inputs.append(a.reshape(n, n))
        truth = execute(inputs, ops, rotations)
        stats = significance(inputs, ops, rotations)
        if acceptable(stats, truth, cfg):
            break
    else:
        raise RuntimeError('Could not generate a non-degenerate composition')
    demos = [demonstrations(op, n, rng) for op in ops]
    question = (
        'Filled cells represent 1 and light cells 0. Each operator '
        + ', '.join('XYZ'[: len(ops)])
        + ' is fixed across its two demonstrations. '
        'Each is one of pixelwise OR, AND, XOR, or left-minus-right: a cell is filled exactly when it is filled on the left and empty on the right. '
        'Different operator labels may represent the same operation. Infer every operator using BOTH of its demonstrations. '
        'The grids labeled I1, I2, I3, I4 as needed are the inputs. Start with I1. '
    )
    for i, turn in enumerate(rotations):
        question += f'Step {i+1}: rotate the current grid {90*turn} degrees clockwise, then apply {"XYZ"[i]} with I{i+2} as the right operand. '
    question += f'Reconstruct the complete final {n} by {n} grid. Return exactly {n} rows of {n} binary digits separated by /, top to bottom and left to right; 1 means filled and 0 means empty.'
    answer = '/'.join(''.join(map(str, row)) for row in truth.tolist())
    return dict(
        id=f'{LABEL}/logic_matrix/{tier}/s{seed}/{index:06d}',
        subtype='logic_matrix',
        difficulty=tier,
        question=question,
        answer=answer,
        ans_type='binary_grid',
        options=[],
        generation=dict(version=LABEL, seed=seed, index=index, rejection_attempts=attempt),
        meta=dict(
            family='joint_demo_composed_boolean',
            n=n,
            operators=ops,
            rotations=rotations,
            demonstrations=demos,
            inputs=[a.tolist() for a in inputs],
            truth=truth.tolist(),
            significance=stats,
            requirements=cfg,
        ),
    )


def render(rec):
    cfg = CONFIG[rec['difficulty']]
    n = cfg['n']
    size = cfg['size']
    im = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(im)
    regions = []
    labels = []

    def text(value, x, y, fs=13):
        font = ImageFont.truetype(str(FONT), fs)
        box = draw.textbbox((x, y), value, font=font)
        assert min(box) >= 0 and box[2] < size and box[3] < size
        draw.text((x, y), value, fill=(25, 30, 40), font=font)
        labels.append(value)

    def grid(a, x, y, cell, name):
        x, y = int(x), int(y)
        tile = n * cell
        box = (x, y, x + tile, y + tile)
        assert x >= 0 and y >= 0 and x + tile < size and y + tile < size
        for prior in regions:
            p = prior['bounds']
            assert min(box[2], p[2]) <= max(box[0], p[0]) or min(box[3], p[3]) <= max(box[1], p[1])
        for r, row in enumerate(a):
            for c, value in enumerate(row):
                draw.rectangle(
                    (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
                    fill=(40, 65, 100) if value else (249, 249, 246),
                    outline=(120, 125, 132),
                    width=1,
                )
        regions.append(dict(name=name, bounds=box, cell=cell, values=a))

    flat = [(op, j, panel) for op, demos in enumerate(rec['meta']['demonstrations']) for j, panel in enumerate(demos)]
    if rec['difficulty'] == 'hard':
        cell = 18
        tile = n * cell
        gap = 24
        left = (size - (3 * tile + 2 * gap)) // 2
        positions = [(left, 8 + k * 84) for k in range(4)]
        input_top = 362
    else:
        cell = 16
        tile = n * cell
        gap = 18
        positions = [(9 + (k % 2) * (size // 2), 30 + (k // 2) * 130) for k in range(6)]
        input_top = 460
    for (op, j, panel), (left, top) in zip(flat, positions):
        for k, a in enumerate(panel):
            grid(a, left + k * (tile + gap), top, cell, f'demo-{op}-{j}-{k}')
        text('XYZ'[op], left + tile + (gap - 10) // 2, top + tile / 2 - 9)
        text('=', left + 2 * tile + gap + (gap - 10) // 2, top + tile / 2 - 9)
    count = len(rec['meta']['inputs'])
    stride = size / count
    for k, a in enumerate(rec['meta']['inputs']):
        left = round(k * stride + (stride - tile) / 2)
        grid(a, left, input_top, cell, f'input-{k}')
        text('I' + str(k + 1), left + tile / 2 - 8, input_top - 23)
    rec['image_labels'] = labels
    rec['render_regions'] = regions
    return im
