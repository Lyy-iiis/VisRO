"""Records and the per-item RNG of the lower-tier puzzle generators (logic_low, mirrors_low,
jigsaw_joint_low, paper_query).  The upper tiers use drawing.rng_for / drawing.record through
builders.build."""

import random
from visro.seeding import seed_text

# fixed RNG namespace: every item of these generators is seeded from it, so it must not change
_RNG_NAMESPACE = seed_text(8314596455237908021)
LABEL = 'puzzles'


def rng_for(task, tier, seed, index):
    return random.Random(f'{_RNG_NAMESPACE}/{task}/{tier}/{seed}/{index}')


def plain(value):
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def record(task, tier, seed, index, question, answer, meta, ans_type='choice'):
    return dict(
        id=f'{LABEL}/{task}/{tier}/s{seed}/{index:08d}',
        subtype=task,
        difficulty=tier,
        question=question,
        answer=answer,
        meta=plain(meta),
        ans_type=ans_type,
        options=list('ABCD') if ans_type == 'choice' else [],
        generation=dict(version=LABEL, seed=seed, index=index),
        image_labels=[],
    )
