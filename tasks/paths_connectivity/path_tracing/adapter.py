"""path_tracing -- follow curves between labelled endpoints.

In-distribution tiers come from the balanced generator (generator/id): blocks of
160 scheduled rows (tier, style, question and appearance quotas); release index k
of a tier is the k-th row of that tier, block by block.  xxhard comes from the
out-of-distribution generator (generator/ood), all five conditions, in its
release schedule order.
"""
from visro import runners
from visro.api import RELEASE_VERSION
from visro.seeding import digest

TASK = 'path_tracing'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_ID = 'k-th tier occurrence in a balanced 160-row block, including styles, questions and appearance quotas.'
TIER_DEFINITIONS = {
    'easy': f'448x448; easy curve_engine parameters; {_ID}',
    'medium': f'448x448; medium curve_engine parameters; {_ID}',
    'hard': f'448x448; hard curve_engine parameters; {_ID}',
    'xhard': f'448x448; xhard curve_engine parameters; {_ID}',
    'xxhard': '448x448; out-of-distribution release_schedule and generate_one, all five conditions.',
}


def _build(tier, seed, split, index):
    if tier == 'xxhard':
        from tasks.paths_connectivity.path_tracing.generator.ood import generate_eval
        total = 40 * ((index + 40) // 40)
        row = generate_eval.release_schedule(total, seed)[index]
        config = {'seed_retries': generate_eval.DEFAULT_SEED_RETRIES}
        record, image = generate_eval.generate_one(row=row, master_seed=seed, size=448,
                                                   seed_retries=generate_eval.DEFAULT_SEED_RETRIES,
                                                   config=config, config_sha256=digest(config))
    else:
        from tasks.paths_connectivity.path_tracing.generator.id import balanced, schedule
        block, position = divmod(index, 40)
        rows = balanced.appearance_schedule(160, schedule.path_schedule_seed(seed, block))
        slots = [(i, row) for i, row in enumerate(rows) if row[1] == tier]
        offset, row = slots[position]
        config = balanced.resolved_config(count=(block + 1) * 160, size=448,
                                          scene_retries=3, geometry_retries=12, seed_retries=8)
        record, image = balanced.generate_one(idx=block * 160 + offset, row=row, seed=seed, size=448,
                                              scene_retries=3, geometry_retries=12, seed_retries=8,
                                              config=config, config_sha256=digest(config))
    return record, image, {}


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index)
