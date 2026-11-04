"""Paper folding with punches at stated stages; integer material-point tracking."""

import math
import numpy as np
from PIL import Image, ImageDraw
from shared.puzzles.drawing import canvas, label, record

PARAMS = {
    'easy': dict(n=6, folds=1, stages=[1], punches=[2], queries=0),
    'medium': dict(n=8, folds=2, stages=[1, 2], punches=[1, 1], queries=0),
    'hard': dict(n=8, folds=3, stages=[1, 3], punches=[1, 2], queries=0),
    'xhard': dict(n=16, folds=4, stages=[2, 3, 4], punches=[1, 1, 2], queries=12),
    'xxhard': dict(n=16, folds=5, stages=[2, 4, 5], punches=[1, 2, 2], queries=16),
}


def fold_plan(n, count, rng, diagonal):
    # All coordinates are doubled: grid-cell centers are odd integers.
    box = [0, 0, 2 * n, 2 * n]
    steps = []
    for k in range(count):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        before = box.copy()
        if diagonal and k == count - 1 and w == h:
            axis = rng.choice(('d', 'a'))
            keep = 'lower'
            line = x0 - y0 if axis == 'd' else x0 + y1
        else:
            axis = 'v' if w > h else 'h' if h > w else rng.choice(('v', 'h'))
            line = (x0 + x1) // 2 if axis == 'v' else (y0 + y1) // 2
            keep = rng.choice(('low', 'high'))
            if axis == 'v':
                box[2 if keep == 'low' else 0] = line
            else:
                box[3 if keep == 'low' else 1] = line
        steps.append(dict(axis=axis, line=line, keep=keep, before=before, after=box.copy()))
    return steps


def advance(points, step):
    out = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    a = step['axis']
    line = step['line']
    if a in ('v', 'h'):
        dim = 0 if a == 'v' else 1
        v = points[:, dim]
        moving = v > line if step['keep'] == 'low' else v < line
        out[moving, dim] = 2 * line - v[moving]
    elif a == 'd':
        moving = x - y > line
        out[moving, 0] = y[moving] + line
        out[moving, 1] = x[moving] - line
    else:
        moving = x + y < line
        out[moving, 0] = line - y[moving]
        out[moving, 1] = line - x[moving]
    return out


def safe_centers(points, step):
    x0, y0, x1, y1 = step['after']
    a = step['axis']
    line = step['line']
    centers = []
    for x, y in sorted({tuple(map(int, p)) for p in points}):
        if not (x0 < x < x1 and y0 < y < y1):
            continue
        if a == 'd' and y - x + line < 1:
            continue
        if a == 'a' and x + y - line < 1:
            continue
        centers.append((x, y))
    return centers


def punched_mask(maps, events, n):
    removed = np.zeros(n * n, dtype=bool)
    for event in events:
        points = maps[event['stage'] - 1]
        for x, y in event['centers']:
            removed |= (points[:, 0] == x) & (points[:, 1] == y)
    return removed.reshape(n, n)


def draw_packet(im, step, n, centers, slot, title, side):
    sx, sy = slot
    label(im, title, (sx, sy), 11)
    box = step['after'] if step else [0, 0, 2 * n, 2 * n]
    x0, y0, x1, y1 = box
    w = (x1 - x0) // 2
    h = (y1 - y0) // 2
    cell = side / max(w, h)
    ox = sx + 24 + (side - w * cell) / 2
    oy = sy + 28 + (side - h * cell) / 2

    def xy(x, y):
        return ox + (x - x0) * cell / 2, oy + (y - y0) * cell / 2

    if step and step['axis'] == 'd':
        vertices = [(x0, y0), (x1, y1), (x0, y1)]
    elif step and step['axis'] == 'a':
        vertices = [(x1, y0), (x1, y1), (x0, y1)]
    else:
        vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    polygon = [xy(x, y) for x, y in vertices]
    mask = Image.new('L', im.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    sheet = Image.new('RGB', im.size, (218, 229, 242))
    d = ImageDraw.Draw(sheet)
    for c in range(w + 1):
        d.line((ox + c * cell, oy, ox + c * cell, oy + h * cell), fill=(155, 174, 195), width=1)
    for r in range(h + 1):
        d.line((ox, oy + r * cell, ox + w * cell, oy + r * cell), fill=(155, 174, 195), width=1)
    im.paste(sheet, (0, 0), mask)
    d = ImageDraw.Draw(im)
    d.line(polygon + [polygon[0]], fill=(45, 60, 78), width=1)
    stride = max(1, math.ceil(12 / cell))

    def ticks(length):
        values = list(range(0, length, stride))
        if values[-1] != length - 1:
            if length - 1 - values[-1] < stride:
                values.pop()
            values.append(length - 1)
        return values

    for c in ticks(w):
        label(im, str(x0 // 2 + c + 1), (ox + (c + 0.5) * cell - 4, oy - 15), 9)
    for r in ticks(h):
        label(im, str(y0 // 2 + r + 1), (ox - 20, oy + (r + 0.5) * cell - 6), 9)
    for x, y in centers:
        px, py = xy(x, y)
        radius = max(2.5, min(5, cell * 0.21))
        d.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(200, 42, 51))
    return dict(min_cell_pixels=cell, punch_radius_pixels=max(2.5, min(5, cell * 0.21)))


def draw_option(im, mask, x, y, side):
    n = len(mask)
    cell = side / n
    d = ImageDraw.Draw(im)
    d.rectangle((x, y, x + side, y + side), fill=(218, 229, 242), outline=(45, 60, 78), width=1)
    for k in range(1, n):
        d.line((x + k * cell, y, x + k * cell, y + side), fill=(169, 184, 202))
        d.line((x, y + k * cell, x + side, y + k * cell), fill=(169, 184, 202))
    radius = max(2.5, cell * 0.21)
    for r, c in zip(*np.nonzero(mask)):
        xx = x + (c + 0.5) * cell
        yy = y + (r + 0.5) * cell
        d.ellipse((xx - radius, yy - radius, xx + radius, yy + radius), fill='white', outline=(45, 60, 78), width=1)
    return [x, y, x + side + 1, y + side + 1]


def paper_folding(tier, rng):
    p = PARAMS[tier]
    n = p['n']
    original = np.array([(2 * c + 1, 2 * r + 1) for r in range(n) for c in range(n)], dtype=int)
    for attempt in range(500):
        diagonal = tier in ('hard', 'xxhard')
        steps = fold_plan(n, p['folds'], rng, diagonal)
        points = original.copy()
        maps = []
        for step in steps:
            points = advance(points, step)
            maps.append(points.copy())
        events = []
        removed = np.zeros((n, n), dtype=bool)
        stage_new = []
        for stage, count in zip(p['stages'], p['punches']):
            legal = safe_centers(maps[stage - 1], steps[stage - 1])
            rng.shuffle(legal)
            chosen = []
            for center in legal:
                candidate = dict(stage=stage, centers=chosen + [center])
                fresh = punched_mask(maps, events + [candidate], n)
                if np.count_nonzero(fresh) > np.count_nonzero(removed):
                    chosen.append(center)
                    removed = fresh
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                break
            prev = punched_mask(maps, events, n)
            events.append(dict(stage=stage, centers=chosen))
            stage_new.append(np.flatnonzero(removed & ~prev).tolist())
        if len(events) != len(p['stages']):
            continue
        truth = punched_mask(maps, events, n)
        options = []
        queries = []
        effects = [punched_mask(maps, [event], n) for event in events]
        exclusive = []
        for i, effect in enumerate(effects):
            rest = np.zeros_like(effect)
            for j, other in enumerate(effects):
                if j != i:
                    rest |= other
            exclusive.append(np.flatnonzero(effect & ~rest).tolist())
        if any(not cells for cells in exclusive):
            continue
        if not p['queries']:
            options = [truth.copy()]
            truth_count = np.count_nonzero(truth)
            for _ in range(400):
                which = rng.randrange(len(events))
                altered = [dict(stage=e['stage'], centers=list(e['centers'])) for e in events]
                centers = altered[which]['centers']
                stage = altered[which]['stage']
                centers[rng.randrange(len(centers))] = rng.choice(safe_centers(maps[stage - 1], steps[stage - 1]))
                candidate = punched_mask(maps, altered, n)
                if np.count_nonzero(candidate) != truth_count:
                    continue
                if min(np.count_nonzero(candidate != o) for o in options) < 4:
                    continue
                options.append(candidate)
                if len(options) == 4:
                    break
            if len(options) != 4:
                continue
            rng.shuffle(options)
            answer = 'ABCD'[next(i for i, o in enumerate(options) if np.array_equal(o, truth))]
        else:
            nq = p['queries']
            ones = np.flatnonzero(truth).tolist()
            zeros = np.flatnonzero(~truth).tolist()
            if len(ones) < nq // 2 or len(zeros) < nq // 2:
                continue
            selected = {rng.choice(indices) for indices in exclusive}
            selected.update(rng.sample([i for i in ones if i not in selected], nq // 2 - len(selected)))
            selected = list(selected) + rng.sample(zeros, nq // 2)
            rng.shuffle(selected)
            queries = [[i // n, i % n] for i in selected]
            answer = ''.join('1' if truth[r, c] else '0' for r, c in queries)
        break
    else:
        raise RuntimeError('Cannot sample an unambiguous staged paper-folding question')
    im = canvas()
    render = dict(packet_panels=[], option_boxes=[])
    # P1/P2/P3 are only identifiers; all fold directions and timing live in q.
    for i, event in enumerate(events):
        if len(events) == 1:
            sx, sy = 118, 22
        elif i < 2:
            sx, sy = 8 + i * 220, 22
        else:
            sx, sy = 118, 224
        metric = draw_packet(im, steps[event['stage'] - 1], n, event['centers'], (sx, sy), 'P' + str(i + 1), 172)
        metric['has_punches'] = True
        render['packet_panels'].append(metric)
    if options:
        for i, o in enumerate(options):
            x = 8 + i * 110
            y = 318
            label(im, 'ABCD'[i], (x + 47, y - 20), 13)
            render['option_boxes'].append(draw_option(im, o, x, y, 104))
    q = f'A square sheet has a {n} by {n} grid. Rows increase downward and columns rightward, both numbered from 1 in the original sheet. Panels {", ".join("P"+str(i+1) for i in range(len(events)))} show the punch stages in order, keeping that fixed coordinate frame and zooming each folded packet separately; grid-square units are unchanged. Red circles mark the centers of small circular punches through ALL layers present at that stage. Earlier holes are not redrawn in later packet diagrams. Fold the WHOLE current packet in the following order, without rotating the diagram: '
    for i, step in enumerate(steps, 1):
        a = step['axis']
        v = step['line'] / 2
        if a == 'v':
            desc = (
                'right half onto left half' if step['keep'] == 'low' else 'left half onto right half'
            ) + f' along the vertical grid line x={v:g}'
        elif a == 'h':
            desc = (
                'bottom half onto top half' if step['keep'] == 'low' else 'top half onto bottom half'
            ) + f' along the horizontal grid line y={v:g}'
        elif a == 'd':
            desc = 'the upper-right triangular half onto the lower-left half along the top-left to bottom-right diagonal of the current square packet'
        else:
            desc = 'the upper-left triangular half onto the lower-right half along the top-right to bottom-left diagonal of the current square packet'
        q += f'Fold {i}: {desc}. '
        if i in p['stages']:
            q += (
                'Immediately make the red punches shown in panel P'
                + str(p['stages'].index(i) + 1)
                + ', then continue. '
            )
    q += 'All listed x/y crease values refer to original grid LINES, with the top-left outer corner at (0,0). Fully unfold the sheet after the final punches. '
    if options:
        q += 'Which bottom option shows exactly all holes? All options show the same number of holes. Return A, B, C, or D.'
    else:
        q += (
            'For these original-sheet cells in the listed order, report whether a hole is centered in that cell: '
            + ', '.join(f'({r+1},{c+1})' for r, c in queries)
            + f'. Return exactly {len(queries)} binary digits, using 1 for a hole and 0 for intact paper; coordinates are (row,column).'
        )
    meta = dict(
        family='staged_paper_punches',
        n=n,
        folds=steps,
        events=events,
        truth=truth.astype(int).tolist(),
        options=[o.astype(int).tolist() for o in options],
        queries=queries,
        render=render,
        attempts=attempt + 1,
    )
    rec = record('paper_folding', tier, q, answer, meta, 'choice' if options else 'exact')
    rec['options'] = list('ABCD') if options else []
    return rec, im
