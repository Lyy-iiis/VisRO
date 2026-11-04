"""Canvas, labels, grids and answer bands shared by the puzzle generators, and the per-item RNG and
record of their upper tiers (see builders.build; runtime.py has the lower-tier ones)."""

from shared.drawing.fonts import font_path
from visro.seeding import seed_text
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

TASKS = ('paper_folding', 'logic_matrix', 'jigsaw_completion', 'mirroring_patterns')
SIZE = 448
MAX_SIZE = 588


def image_size(task, tier):
    return MAX_SIZE if task in ('logic_matrix', 'jigsaw_completion') and tier in ('xhard', 'xxhard') else SIZE


COLORS = [(249, 249, 246), (199, 49, 58), (42, 103, 193), (40, 151, 99), (231, 182, 37), (142, 78, 172)]
CODES = '.RBGYP'

# fixed RNG namespace: every item of these generators is seeded from it, so it must not change
_RNG_NAMESPACE = seed_text(544942431748362718770739)


def rng_for(task, tier, seed, index):
    return random.Random(f'{_RNG_NAMESPACE}/{task}/{tier}/{seed}/{index}')


def font(size=14):
    return ImageFont.truetype(str(font_path('sans')), size)


def canvas(size=SIZE):
    return Image.new('RGB', (size, size), 'white')


def label(im, text, xy, size=14, color=(25, 30, 40)):
    d = ImageDraw.Draw(im)
    f = font(size)
    box = d.textbbox(xy, text, font=f)
    if box[0] < 0 or box[1] < 0 or box[2] > im.width or box[3] > im.height:
        raise RuntimeError(f'Text exceeds image bounds: {text!r} {box}')
    d.text(xy, text, fill=color, font=f)
    im.info['diagram_labels'] = [*im.info.get('diagram_labels', []), str(text)]


def draw_grid(im, grid, x, y, cell, coords=False, binary=False):
    d = ImageDraw.Draw(im)
    a = np.asarray(grid)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            value = int(a[r, c])
            fill = ((40, 65, 100) if value else (249, 249, 246)) if binary else COLORS[value]
            d.rectangle(
                (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
                fill=fill,
                outline=(120, 125, 132),
                width=1,
            )
    if coords:
        for k in range(a.shape[0]):
            label(im, str(k + 1), (x - 22, y + k * cell + cell / 2 - 7), 12)
        for k in range(a.shape[1]):
            label(im, str(k + 1), (x + k * cell + cell / 2 - 7, y - 20), 12)


def record(task, tier, question, answer, meta, answer_kind='exact'):
    return dict(
        subtype=task,
        difficulty=tier,
        question=question,
        answer=answer,
        ans_type=answer_kind,
        options=[],
        meta=meta,
    )
