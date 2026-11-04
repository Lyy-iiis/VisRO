"""maze -- reachability, distance-order and radius questions on a rendered maze.

The generator (generator/maze_core.py) builds blocks of four items, one per
in-distribution tier (form, lattice and rim schedules); release index k of a tier
is block k at that tier's position, xxhard draws block k of the
out-of-distribution split.  The generator's pre-render validation returns the
accepted maze graph so the verifier can recompute the answer.
"""
from visro import runners
from visro.api import RELEASE_VERSION

TASK = 'maze'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_RECIPE = 'from maze_core.block_record, with its form/lattice/rim schedule and border fix;'
TIER_DEFINITIONS = {
    'easy': f'448x448; easy {_RECIPE} block=4*index+tier_position.',
    'medium': f'448x448; medium {_RECIPE} block=4*index+tier_position.',
    'hard': f'448x448; hard {_RECIPE} block=4*index+tier_position.',
    'xhard': f'448x448; xhard {_RECIPE} block=4*index+tier_position.',
    'xxhard': f'448x448; xxhard {_RECIPE} out-of-distribution block=index.',
}


def _build(tier, seed, split, index):
    from tasks.paths_connectivity.maze.generator import maze_core, questions
    role = 'ood_eval' if tier == 'xxhard' else 'train_id'
    gen_index = index if tier == 'xxhard' else 4 * index + runners.ID_TIERS.index(tier)
    palettes = questions.resolve_palettes('eval' if role == 'ood_eval' else 'train')
    record, image, audit = maze_core.block_record(seed, gen_index, role, runners.generator_split(tier, split), palettes)
    return record, image, audit


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index)
