"""Easy/medium logic_matrix: the matrix_base images with explicit, exhaustible rule families."""

import numpy as np
from PIL import ImageDraw
from shared.puzzles.runtime import record, rng_for, plain
from . import matrix_base
from .attribute_rules import logic_question, check_logic_low


def build(tier, seed, index):
    src = matrix_base
    rng = rng_for('logic-low', tier, seed, index)
    kind = 'attribute' if index % 2 == 0 else 'overlay'
    for attempt in range(100):
        prob = src.attr_progression_problem(rng, tier) if kind == 'attribute' else src.overlay_problem(rng, tier)
        answer, ok = src.verify_unique(prob)
        if not ok:
            continue
        family = 'explicit_' + kind
        if kind == 'attribute':
            m = dict(
                family=family,
                rows=prob['rows'],
                cols=prob['cols'],
                changing_attributes=3 if tier == 'easy' else 4,
                grid_attr=plain(prob['grid']),
                options=[dict(attr=plain(x)) for x, _, _ in prob['options']],
            )
        else:
            m = dict(
                family=family,
                grid_kind=prob['grid_kind'],
                R=prob['R'],
                C=prob['C'],
                row_masks=plain(prob['rows']),
                options=[dict(mask=plain(x)) for x, _, _ in prob['options']],
                on_color=plain(prob['on_color']),
            )
        m['attempts'] = attempt + 1
        rec = record('logic_matrix', tier, seed, index, logic_question(tier, family), answer, m)
        try:
            check_logic_low(rec)
        except AssertionError:
            continue
        return rec
    raise RuntimeError('Could not generate a uniquely solved explicit-rule matrix')


def to_problem(rec):
    m = rec['meta']
    kind = m['family'].removeprefix('explicit_')
    glyph = lambda a: dict(a, color=tuple(a['color']))
    if kind == 'attribute':
        return dict(
            family=kind,
            rows=m['rows'],
            cols=m['cols'],
            grid=[[glyph(a) for a in row] for row in m['grid_attr']],
            answer_cell=glyph(m['grid_attr'][-1][-1]),
            options=[(glyph(o['attr']), i == 'ABCD'.index(rec['answer']), '') for i, o in enumerate(m['options'])],
        )
    return dict(
        family=kind,
        grid_kind=m['grid_kind'],
        R=m['R'],
        C=m['C'],
        rows=[[np.asarray(x, dtype=int) for x in row] for row in m['row_masks']],
        answer_mask=np.asarray(m['row_masks'][1][2]),
        options=[(np.asarray(o['mask']), i == 'ABCD'.index(rec['answer']), '') for i, o in enumerate(m['options'])],
        on_color=tuple(m['on_color']),
    )


def render(rec):
    src = matrix_base
    prob = to_problem(rec)
    original = ImageDraw.ImageDraw.text

    def tracked(draw, xy, text, *args, **kwargs):
        draw._image.info['diagram_labels'] = [*draw._image.info.get('diagram_labels', []), str(text)]
        return original(draw, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = tracked
    try:
        im = src.render_problem_image(prob, 448)
    finally:
        ImageDraw.ImageDraw.text = original
    rec['image_labels'] = im.info.get('diagram_labels', [])
    return im
