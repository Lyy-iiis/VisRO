"""overlapping_glyphs -- read the characters of overlapping glyphs in the order of their colour boxes.

The generator (generator/glyph_runner.py) writes shards from its own index: mode,
outline, subset and partition schedules interleave the four in-distribution
tiers; release index k of a tier is the k-th index of that tier, xxhard draws the
out-of-distribution split.  With VISRO_TOKENIZER_DIR set (docs/ASSETS.md) the
runner also applies its tokenizer answer filter.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from visro import runners
from visro.api import AssetError, RELEASE_VERSION

TASK = 'overlapping_glyphs'
VERSION = f'{TASK}@{RELEASE_VERSION}'
_RECIPE = 'mode, outline, subset and partition schedules; optional pinned tokenizer filter.'
TIER_DEFINITIONS = {
    'easy': f'448x448; 3 glyphs, one group; {_RECIPE}',
    'medium': f'448x448; 4 glyphs, one group; {_RECIPE}',
    'hard': f'448x448; 5 glyphs, partitions 5 / 3+2 / 1+4; {_RECIPE}',
    'xhard': f'448x448; 6 glyphs, partitions 6 / 3+3 / 2+4 / 1+5; {_RECIPE}',
    'xxhard': f'448x448; 5-8 glyphs, 1-4 feasible groups, max 6 per group; out-of-distribution materials; {_RECIPE}',
}
_TOKENIZER_SHA256 = '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'


def _build(tier, seed, split, index):
    from tasks.discrimination_grouping.overlapping_glyphs.generator import glyph_runner

    gen_split = runners.generator_split(tier, split, validation_name=True)
    total = glyph_runner.TOTALS[gen_split]
    gen_index = index % total if tier == 'xxhard' else 4 * (index % (total // 4)) + runners.ID_TIERS.index(tier)
    # The runner supports an optional tokenizer.  Keep that choice in the record;
    # only runs with the pinned model claim its token filter.
    directory = os.environ.get('VISRO_TOKENIZER_DIR')
    tokenizer = Path(directory) / 'paligemma_tokenizer.model' if directory else None
    if tokenizer is not None:
        if not tokenizer.is_file() or hashlib.sha256(tokenizer.read_bytes()).hexdigest() != _TOKENIZER_SHA256:
            raise AssetError('glyph tokenizer does not match its pinned hash; see docs/ASSETS.md')
    with tempfile.TemporaryDirectory(prefix='visro-glyph-') as tmp:
        root = Path(tmp)
        with redirect_stdout(io.StringIO()):
            glyph_runner.run_shard(root, seed, gen_split, total, gen_index, 1, tokenizer_path=tokenizer)
        record = json.loads((root / 'records.jsonl').read_text())
        return record, runners.read_image(root, record['image']), {'tokenizer_filter': tokenizer is not None,
                                                                  'total': total}


def build(tier, seed, *, split, index):
    return runners.build(TASK, _build, tier, seed, split=split, index=index)
