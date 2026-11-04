"""flow_free -- questions about the pipes of a solved Flow Free board.

The generator (generator/production.py) schedules every item from its own index
in blocks of 128 (tier, question form, style and materials).  Release index k of
a tier maps to the k-th occurrence of that tier in the schedule; xxhard draws the
out-of-distribution split (five strata, in index order).
"""
from functools import lru_cache

from visro import runners
from visro.api import RELEASE_VERSION

TASK = 'flow_free'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_SCHEDULE = 'split-aware schedule and materials; k-th occurrence of this tier in the 128-record schedule.'
TIER_DEFINITIONS = {
    'easy': f'448x448; easy from production.generate_one; {_SCHEDULE}',
    'medium': f'448x448; medium from production.generate_one; {_SCHEDULE}',
    'hard': f'448x448; hard from production.generate_one; {_SCHEDULE}',
    'xhard': f'448x448; xhard from production.generate_one; {_SCHEDULE}',
    'xxhard': '448x448; xxhard from production.generate_one; split-aware schedule and materials; '
              'five out-of-distribution strata, in index order.',
}


@lru_cache(maxsize=12)
def _slots(tier, gen_split):
    from tasks.paths_connectivity.flow_free.generator import board_core, production
    role = production.ROLE_BY_SPLIT[gen_split]
    return tuple(i for i in range(128) if board_core.schedule(i, role).tier == tier)


def _build(tier, seed, split, index):
    from tasks.paths_connectivity.flow_free.generator import production
    gen_split = runners.generator_split(tier, split)
    if tier == 'xxhard':
        gen_index = index
    else:
        slots = _slots(tier, gen_split)
        cycle, position = divmod(index, len(slots))
        gen_index = cycle * 128 + slots[position]
    total = max(production.TOTALS[gen_split], 20 * ((gen_index + 20) // 20))
    record, image = production.generate_one(seed, gen_index, gen_split, total=total)
    return record, image, {'total': total}


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index)
