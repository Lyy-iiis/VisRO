"""easy/medium/hard paper folding in the xhard/xxhard query form: report a hole bitstring.

The physical puzzle follows paper_low.CONFIG (sheet size, fold depths, punch stages);
pool() enumerates at least 25k cores per tier.  The answer is the bitstring of
6/8/10 queried cells, against 12/16 on the xhard/xxhard tiers (folding.py).
"""

import copy, itertools, math
from functools import lru_cache
import numpy as np
from shared.puzzles import drawing as draw
from . import folding
from .paper_low import all_plans
from shared.puzzles.runtime import record, rng_for

# Geometry identical to paper_low.CONFIG (see GEOMETRY assertion below); only
# 'queries' is added.  Ladder against the folding.py tiers
# (xhard n=16/folds=4/stages=3/queries=12, xxhard n=16/folds=5/stages=3/queries=16):
#   n       8    8    8   < 16   16
#   folds   1    2    3   <  4    5
#   stages  1    2    2   <  3    3
#   queries 6    8   10   < 12   16
CONFIG = {
    'easy': dict(n=8, folds=1, stages=[1], punches=[4], queries=6),
    'medium': dict(n=8, folds=2, stages=[1, 2], punches=[2, 2], queries=8),
    'hard': dict(n=8, folds=3, stages=[1, 3], punches=[2, 2], queries=10),
}
GEOMETRY = ('n', 'folds', 'stages', 'punches')


def _diagonal(tier):
    # Matches paper_low and folding.py PARAMS: the last fold may be a diagonal at hard.
    return tier == 'hard'


@lru_cache(None)
def pool(tier):
    """Every legal (fold plan, punch assignment) for the tier, in a fixed order."""
    cfg = CONFIG[tier]
    n = cfg['n']
    plans = all_plans(n, cfg['folds'], _diagonal(tier))
    blocks = []
    total = 0
    for steps in plans:
        pts = np.array([(2 * c + 1, 2 * r + 1) for r in range(n) for c in range(n)])
        maps = []
        for step in steps:
            pts = folding.advance(pts, step)
            maps.append(pts.copy())
        domains = []
        legal_by_stage = []
        mask_by_stage = []
        for stage, count in zip(cfg['stages'], cfg['punches']):
            legal = folding.safe_centers(maps[stage - 1], steps[stage - 1])
            legal_by_stage.append(legal)
            masks = {c: sum(1 << i for i, q in enumerate(maps[stage - 1]) if tuple(q) == c) for c in legal}
            mask_by_stage.append(masks)
            choices = []
            for centers in itertools.combinations(legal, count):
                mask = 0
                for c in centers:
                    mask |= masks[c]
                choices.append((centers, mask))
            domains.append(choices)
        accepted = []
        for indices in itertools.product(*(range(len(d)) for d in domains)):
            effects = [domains[i][j][1] for i, j in enumerate(indices)]
            valid = True
            prior = 0
            for i, idx in enumerate(indices):
                current = prior
                for c in domains[i][idx][0]:
                    if not mask_by_stage[i][c] & ~current:
                        valid = False
                        break
                    current |= mask_by_stage[i][c]
                prior = current
                rest = 0
                for j, effect in enumerate(effects):
                    if i != j:
                        rest |= effect
                if not effects[i] & ~rest:
                    valid = False
            if valid:
                accepted.append(indices)
        ranks = np.asarray(accepted, dtype=np.int32)
        total += len(ranks)
        blocks.append(
            dict(
                steps=steps,
                maps=maps,
                domains=domains,
                legal=legal_by_stage,
                masks=mask_by_stage,
                ranks=ranks,
                end=total,
            )
        )
    return blocks, total


@lru_cache(None)
def permutation(tier, seed):
    _, n = pool(tier)
    rng = rng_for('paper-query-order', tier, seed, 0)
    a = rng.randrange(1, n)
    while math.gcd(a, n) != 1:
        a = (a + 1) % n or 1
    return a, rng.randrange(n)


def question(meta):
    n = meta['n']
    events = meta['events']
    queries = meta['queries']
    q = f'A square sheet has an {n} by {n} grid. Rows increase downward and columns rightward, numbered from 1 in the original sheet. '
    q += (
        ('Panels ' if len(events) > 1 else 'Panel ')
        + ', '.join('P' + str(i + 1) for i in range(len(events)))
        + (' show' if len(events) > 1 else ' shows')
        + ' successive punch stages in the same original-sheet coordinate frame; each packet is zoomed separately and the grid-square units are unchanged. Red circles are small punches through ALL current layers. Earlier holes are not redrawn. Fold the WHOLE packet in this order without rotating the diagram: '
    )
    stages = [e['stage'] for e in events]
    for i, s in enumerate(meta['folds'], 1):
        axis = s['axis']
        line = s['line'] / 2
        if axis == 'v':
            desc = (
                'right half onto left half' if s['keep'] == 'low' else 'left half onto right half'
            ) + f' along the vertical grid line x={line:g}'
        elif axis == 'h':
            desc = (
                'bottom half onto top half' if s['keep'] == 'low' else 'top half onto bottom half'
            ) + f' along the horizontal grid line y={line:g}'
        elif axis == 'd':
            desc = (
                'upper-right half onto lower-left half along the top-left to bottom-right diagonal of the square packet'
            )
        else:
            desc = (
                'upper-left half onto lower-right half along the top-right to bottom-left diagonal of the square packet'
            )
        q += f'Fold {i}: {desc}. '
        if i in stages:
            q += 'Immediately make the punches in panel P' + str(stages.index(i) + 1) + '. '
    q += 'Crease coordinates refer to original grid LINES, with the top-left outer corner at (0,0). Fully unfold after all punches. '
    q += 'For these original-sheet cells in the listed order, report whether a hole is centered in that cell: '
    q += ', '.join(f'({r+1},{c+1})' for r, c in queries)
    q += f'. Return exactly {len(queries)} binary digits, using 1 for a hole and 0 for intact paper; coordinates are (row,column).'
    return q


def build(tier, seed, index):
    blocks, total = pool(tier)
    if index >= total:
        raise ValueError('Paper core index exceeds its finite capacity')
    a, b = permutation(tier, seed)
    rank = (a * index + b) % total
    previous = 0
    for block in blocks:
        if rank < block['end']:
            break
        previous = block['end']
    cfg = CONFIG[tier]
    n = cfg['n']
    ids = block['ranks'][rank - previous]
    events = [
        dict(stage=s, centers=[list(c) for c in block['domains'][i][int(j)][0]])
        for i, (s, j) in enumerate(zip(cfg['stages'], ids))
    ]
    effects = []
    for i, event in enumerate(events):
        mask = 0
        for c in event['centers']:
            mask |= block['masks'][i][tuple(c)]
        effects.append(mask)
    truth = 0
    for mask in effects:
        truth |= mask
    # Every stage owns cells no other stage punches (pool() enforces it); querying
    # one of each keeps the bitstring sensitive to every stage of the plan.
    exclusive = []
    for i, mask in enumerate(effects):
        rest = 0
        for j, other in enumerate(effects):
            if i != j:
                rest |= other
        own = mask & ~rest
        cells = [k for k in range(n * n) if own >> k & 1]
        if not cells:
            raise RuntimeError('Punch stage without exclusive holes escaped the pool filter')
        exclusive.append(cells)
    nq = cfg['queries']
    rng = rng_for('paper-query-cells', tier, seed, index)
    picked = [rng.choice(cells) for cells in exclusive]
    if len(picked) > nq:
        raise RuntimeError('Too few queries to cover every punch stage')
    taken = set(picked)
    ones = [k for k in range(n * n) if truth >> k & 1 and k not in taken]
    zeros = [k for k in range(n * n) if not truth >> k & 1]
    extra = max(0, min(nq // 2 - len(picked), len(ones)))
    picked += rng.sample(ones, extra)
    need = nq - len(picked)
    if need > len(zeros):
        raise RuntimeError('Not enough intact cells to fill the query list')
    picked += rng.sample(zeros, need)
    rng.shuffle(picked)
    queries = [[k // n, k % n] for k in picked]
    answer = ''.join('1' if truth >> k & 1 else '0' for k in picked)
    grid = lambda bits: [[int(bool(bits & (1 << (r * n + c)))) for c in range(n)] for r in range(n)]
    m = dict(
        family='staged_paper_punches',
        n=n,
        folds=copy.deepcopy(block['steps']),
        events=events,
        truth=grid(truth),
        options=[],
        queries=queries,
        requirements=cfg,
        core_rank=rank,
        core_capacity=total,
    )
    rec = record('paper_folding', tier, seed, index, question(m), answer, m, 'exact')
    rec['builder_family'] = 'staged_paper_punches'
    return rec


PUNCH = (200, 42, 51)


def panel_slot(count, i):
    # No option strip below (the answer is a bitstring), so the packet row is
    # centred in the 448px canvas instead of being pinned under a legend.
    return (118, 130) if count == 1 else (8 + i * 220, 130)


def verify_punch_pixels(rec, im, side=172):
    """The query form has no option grids, so oracles.check_render never inspects
    pixels.  Assert instead that every red punch dot really sits at the packet
    coordinate the record claims."""
    m = rec['meta']
    for i, event in enumerate(m['events']):
        step = m['folds'][event['stage'] - 1]
        sx, sy = panel_slot(len(m['events']), i)
        x0, y0, x1, y1 = step['after']
        w = (x1 - x0) // 2
        h = (y1 - y0) // 2
        cell = side / max(w, h)
        ox = sx + 24 + (side - w * cell) / 2
        oy = sy + 28 + (side - h * cell) / 2
        for x, y in event['centers']:
            px = int(round(ox + (x - x0) * cell / 2))
            py = int(round(oy + (y - y0) * cell / 2))
            if im.getpixel((px, py)) != PUNCH:
                raise AssertionError(f'Punch {(x,y)} missing from panel P{i+1}')


def render(rec):
    m = rec['meta']
    im = draw.canvas()
    metrics = dict(packet_panels=[], option_boxes=[])
    for i, event in enumerate(m['events']):
        sx, sy = panel_slot(len(m['events']), i)
        metric = folding.draw_packet(
            im, m['folds'][event['stage'] - 1], m['n'], event['centers'], (sx, sy), 'P' + str(i + 1), 172
        )
        metric['has_punches'] = True
        metrics['packet_panels'].append(metric)
    m['render'] = metrics
    rec['image_labels'] = im.info.get('diagram_labels', [])
    verify_punch_pixels(rec, im)
    return im


def _check_geometry():
    from . import paper_low

    for tier, cfg in CONFIG.items():
        base = paper_low.CONFIG[tier]
        if {k: cfg[k] for k in GEOMETRY} != {k: base[k] for k in GEOMETRY}:
            raise RuntimeError('paper_query geometry drifted from paper_low: ' + tier)


_check_geometry()
