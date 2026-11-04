"""Generate glyph shards with exact split/difficulty and nuisance budgets."""

from __future__ import annotations
from visro.paths import ROOT as _ROOT
import collections
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import time

for env in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(env, '1')

# after the thread limits: glyph_core loads NumPy
from . import glyph_core, partition_policy  # noqa: E402

TIERS = ('easy', 'medium', 'hard', 'xhard')
TOTALS = {'train': 100000, 'eval': 600, 'validation': 100, 'ood': 600}
VERSION = 'glyph_dataset_runner'


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(glyph_core.canonical(parts).encode()).digest()[:8], 'big')


@lru_cache(maxsize=32)
def schedule(seed, split, total):
    tiers = ('xxhard',) if split == 'ood' else TIERS
    if total <= 0 or total % len(tiers):
        raise ValueError('total must divide evenly across split difficulties')
    n = total // len(tiers)
    result = {}
    # Integer budgets are allocated across tiers so overall totals are exact.
    for feature, denominator in [('outline', 4), ('subset', 2), ('mode', 2)]:
        target = round(total / denominator)
        base = n // denominator
        extra = target - base * len(tiers)
        ranks = list(tiers)
        random.Random(seed_for(seed, split, feature, 'rounding')).shuffle(ranks)
        for tier in tiers:
            count = base + int(ranks.index(tier) < extra)
            values = [True] * count + [False] * (n - count)
            random.Random(seed_for(seed, split, tier, feature, 'schedule')).shuffle(values)
            result[tier, feature] = values
    return result


def condition(seed, split, total, index):
    if not 0 <= index < total:
        raise ValueError('record index outside split plan')
    tiers = ('xxhard',) if split == 'ood' else TIERS
    tier = tiers[index % len(tiers)]
    ordinal = index // len(tiers)
    plan = schedule(seed, split, total)
    return dict(
        tier=tier,
        mode='global' if plan[tier, 'mode'][ordinal] else 'cyclic',
        outline=plan[tier, 'outline'][ordinal],
        subset=plan[tier, 'subset'][ordinal],
    )


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def existing_records(path):
    """Recover only an interrupted final JSONL write; corruption earlier is fatal."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    offset = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'corrupt record at line {i+1}')
            with path.open('r+b') as handle:
                handle.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    if lines and rows and offset == len(data) and not data.endswith(b'\n'):
        with path.open('ab') as handle:
            handle.write(b'\n')
    return rows


def validate_meta(meta, cond):
    assert meta['tier'] == cond['tier'] and meta['mode'] == cond['mode']
    assert meta['outline'] == cond['outline']
    ids = meta['query_ids']
    glyphs = meta['glyphs']
    assert len(ids) == meta['n_queried'] and len(set(ids)) == len(ids)
    assert sum(meta['groups']) == len(glyphs) == meta['n_characters'] <= 8
    assert max(meta['groups']) <= 6
    colors = [glyphs[i]['color'] for i in ids]
    assert len({g['color'] for g in glyphs}) == len(glyphs)
    assert colors == meta['query_colors'] == meta['swatch_order']
    assert len(meta['swatch_boxes']) == len(ids)
    assert meta['answer'] == ','.join(glyphs[i]['char'] for i in ids)
    assert (len(ids) < len(glyphs)) == cond['subset']


def wrap_record(index, candidate, seed, split, meta, pixel_hash, png_hash, config_hash, failures):
    tier = meta.get('tier', meta.get('difficulty'))
    answer = meta['answer']
    image = f'images/{index:08d}.png'
    return dict(
        id=f'overlapping_glyphs/{split}/s{seed}/{index:08d}',
        index=index,
        task='overlapping_glyphs',
        task_type='visual_tracking',
        subtype='recognize_numbers_and_letters',
        level='fine',
        split=split,
        difficulty=tier,
        difficulty_rank=(*TIERS, 'xxhard').index(tier) + 1,
        image=image,
        question=meta['question'],
        question_style='read_requested_color_boxes',
        answer=answer,
        ans_type='blank',
        options=[],
        qa=dict(
            answer_type='blank',
            answer_text=answer,
            structured_answer=answer,
            options=[],
            grading=dict(method='exact_string', version='1', parameters={}),
        ),
        generation=dict(
            generator_version=VERSION,
            seed=seed,
            index=index,
            candidate_index=candidate,
            config_sha256=config_hash,
            candidate_rejections=failures,
        ),
        pixel_sha256=pixel_hash,
        png_sha256=png_hash,
        meta=meta,
    )


def run_shard(
    out,
    seed,
    split,
    total,
    start,
    count,
    max_retries=40,
    tokenizer_path=None,
):
    if count <= 0 or start < 0 or start + count > total:
        raise ValueError('invalid shard range')
    tokenizer = None
    if tokenizer_path:
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    config = dict(
        version=VERSION,
        runner_sha256=hashlib.sha256(
            (_ROOT / 'tasks/discrimination_grouping/overlapping_glyphs/generator/glyph_runner.py').read_bytes()
        ).hexdigest(),
        seed=seed,
        split=split,
        total=total,
        start=start,
        count=count,
        outline_fraction=0.25,
        subset_fraction=0.5,
        source_sha256=glyph_core.source_fingerprint()[1],
        max_retries=max_retries,
        tokenizer_sha256=hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest() if tokenizer_path else None,
    )
    config_hash = hashlib.sha256(glyph_core.canonical(config).encode()).hexdigest()
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('existing shard configuration differs')
    else:
        atomic_json(out / 'config.json', config)
    rows = existing_records(out / 'records.jsonl')
    for ordinal, row in enumerate(rows):
        if row['index'] != start + ordinal:
            raise ValueError('resume prefix is not contiguous')
        if row['generation']['seed'] != seed or row['generation']['config_sha256'] != config_hash:
            raise ValueError('saved resume record provenance differs')
        expected = condition(seed, split, total, row['index'])
        validate_meta(row['meta'], expected)
        if row['meta']['groups'] != partition_policy.sample_partition(
            expected['tier'], seed_for(seed, split, row['index'], 'planned_partition')
        ):
            raise ValueError('saved resume partition differs')
        image = out / row['image']
        if not image.exists() or hashlib.sha256(image.read_bytes()).hexdigest() != row['png_sha256']:
            raise ValueError('saved resume image missing or corrupted')
    if len(rows) > count:
        raise ValueError('too many records in existing shard')
    clock = time.monotonic()
    previous = len(rows)
    with (out / 'records.jsonl').open('a') as handle:
        for index in range(start + len(rows), start + count):
            cond = condition(seed, split, total, index)
            groups = partition_policy.sample_partition(cond['tier'], seed_for(seed, split, index, 'planned_partition'))
            failures = []
            for attempt in range(max_retries):
                candidate = index + attempt * 1_000_000
                try:
                    image, meta = glyph_core.make_sample(seed, candidate, split=split, groups=groups, **cond)
                    if tokenizer:
                        length = len(tokenizer.encode(meta['answer'])) + 1
                        if length > 16:
                            failures.append(
                                dict(
                                    reason='answer_token_limit', candidate_index=candidate, tokens_including_eos=length
                                )
                            )
                            continue
                        meta['answer_tokens_including_eos'] = length
                    break
                except glyph_core.GenerationRejected as exc:
                    failures.append(getattr(exc, 'metadata', {'reason': str(exc)}))
            else:
                atomic_json(out / 'failure.json', dict(index=index, condition=cond, groups=groups, rejections=failures))
                raise RuntimeError(f'candidate retries exhausted at {split}/{index}')
            validate_meta(meta, cond)
            assert meta['groups'] == groups
            assert image.mode == 'RGB' and image.size == (448, 448)
            path = out / 'images' / f'{index:08d}.png'
            temporary = path.with_suffix('.png.tmp')
            image.save(temporary, format='PNG')
            temporary.replace(path)
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            png_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            row = wrap_record(index, candidate, seed, split, meta, pixel_hash, png_hash, config_hash, failures)
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == count:
                status = dict(
                    split=split,
                    start=start,
                    accepted=len(rows),
                    requested=count,
                    elapsed_seconds=time.monotonic() - clock,
                    complete=len(rows) == count,
                    newly_generated=len(rows) - previous,
                )
                atomic_json(out / 'status.json', status)
                print(json.dumps(status), flush=True)
    status = dict(
        split=split,
        start=start,
        accepted=len(rows),
        requested=count,
        complete=True,
        elapsed_seconds=time.monotonic() - clock,
        newly_generated=len(rows) - previous,
        difficulties=dict(collections.Counter(r['difficulty'] for r in rows)),
        subset_count=sum(r['meta']['n_queried'] < r['meta']['n_characters'] for r in rows),
        outline_count=sum(bool(r['meta']['outline']) for r in rows),
    )
    atomic_json(out / 'status.json', status)
    return status
