"""metro_map -- stops, transfers, station counts and nearest-ring questions on a metro map.

The generator (generator/metro_core.py) builds blocks of four items, one per
in-distribution tier (route rules, styles and form schedule), one question per
image; release index k of a tier is block k at that tier's position, xxhard draws
block k of the out-of-distribution split.
"""
from visro import runners
from visro.api import RELEASE_VERSION

TASK = 'metro_map'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_RECIPE = 'from metro_core.block_records with questions_per_image=1; route rules, styles and form schedule;'
TIER_DEFINITIONS = {
    'easy': f'448x448; easy {_RECIPE} block=4*index+tier_position.',
    'medium': f'448x448; medium {_RECIPE} block=4*index+tier_position.',
    'hard': f'448x448; hard {_RECIPE} block=4*index+tier_position.',
    'xhard': f'448x448; xhard {_RECIPE} block=4*index+tier_position.',
    'xxhard': f'448x448; xxhard {_RECIPE} out-of-distribution block=index.',
}


def _build(tier, seed, split, index):
    from tasks.paths_connectivity.metro_map.generator import metro_core
    role = 'ood_eval' if tier == 'xxhard' else 'train_id'
    gen_index = index if tier == 'xxhard' else 4 * index + runners.ID_TIERS.index(tier)
    records, image = metro_core.block_records(seed, gen_index, role, split=runners.generator_split(tier, split),
                                             questions_per_image=1)
    if len(records) != 1:
        raise ValueError('expected one question per image')
    return records[0], image, {}


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index)
