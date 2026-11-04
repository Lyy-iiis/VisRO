"""count_clusters -- count the objects a question predicate selects in a cluttered scene.

The generator (generator/production.py) schedules every item from its own index:
`spec_for_index` picks the tier, the question pool (total / attribute / region /
overlap / category), the material and the layout.  Release index k of a tier maps
to the k-th scheduled slot of that tier; xxhard draws the out-of-distribution
split.  Material `fluent` / `mixed` scenes need the prepared Fluent Emoji pack
(docs/ASSETS.md).
"""
from functools import lru_cache
from pathlib import Path
import tempfile

from visro import runners
from visro.api import AssetError, RELEASE_VERSION

TASK = 'count_clusters'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_POOLS = 'five question pools total/attribute/region/overlap/category; spec_for_index scheduling.'
TIER_DEFINITIONS = {
    'easy': f'448x448; 8-12 objects; {_POOLS}',
    'medium': f'448x448; 13-20 objects; {_POOLS}',
    'hard': f'448x448; 21-28 objects; {_POOLS}',
    'xhard': f'448x448; 29-38 objects; {_POOLS}',
    'xxhard': f'448x448; standard 39-48 or extended 52-64 objects; {_POOLS}',
}


@lru_cache(maxsize=1)
def _assets():
    from tasks.discrimination_grouping.count_clusters.generator import production
    try:
        return production.verify_fluent_assets()
    except (OSError, ValueError) as exc:
        raise AssetError(f'count_clusters requires the prepared Fluent PNG pack; see docs/ASSETS.md: {exc}') from exc


def _build(tier, seed, split, index):
    from tasks.discrimination_grouping.count_clusters.generator import production
    gen_split = runners.generator_split(tier, split)
    gen_index = index if tier == 'xxhard' else 20 * (index // 5) + 5 * runners.ID_TIERS.index(tier) + index % 5
    spec = production.spec_for_index(gen_index, split=gen_split, seed=seed)
    if spec['material'] in ('fluent', 'mixed'):
        _assets()
    with tempfile.TemporaryDirectory(prefix='visro-count-') as tmp:
        root = Path(tmp)
        record = production.generate_record(gen_index, split=gen_split, seed=seed, out=root)
        return record, runners.read_image(root, record['image']), {}


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index, answer_type='int')
