"""One upper-tier puzzle record: the task's builder on the upper-tier per-item RNG (drawing.rng_for)."""

from shared.puzzles.drawing import MAX_SIZE, TASKS, rng_for
from shared.puzzles.runtime import LABEL


def build(task, tier, seed, index, builder):
    if task not in TASKS:
        raise ValueError(f'unknown puzzles task: {task}')
    rng = rng_for(task, tier, seed, index)
    rec, im = builder(tier, rng)
    rec['builder_family'] = rec['meta']['family']
    rec.update(
        id=f'{LABEL}/{task}/{tier}/s{seed}/{index:06d}',
        difficulty=tier,
        generation=dict(version=LABEL, seed=seed, index=index, image_size=im.width, max_image_size=MAX_SIZE),
        image_labels=im.info.get('diagram_labels', []),
    )
    return rec, im
