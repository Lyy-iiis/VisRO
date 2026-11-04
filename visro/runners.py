"""Helpers for task generators that run as self-contained record writers.

Some generators write a complete record (question, answer, image) for one
scheduled index into a directory of their own.  These helpers read back what
such a generator wrote and wrap it as an Item.
"""
from __future__ import annotations

import hashlib

from visro import api
from visro.seeding import digest, plain

ID_TIERS = api.TIERS[:4]


def generator_split(tier, split, *, validation_name=False):
    """The generator's own split for a (tier, split) request: xxhard always draws the out-of-distribution
    split, a requested ood split of the other tiers draws eval."""
    value = 'ood' if tier == 'xxhard' else ('eval' if split == 'ood' else split)
    return 'validation' if validation_name and value == 'val' else value


def pixel_hash(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def answer(record):
    value = record.get('answer_text', record.get('answer'))
    return str(value['text'] if isinstance(value, dict) else value)


def key(task, image_hash, question):
    # Exact rendered question identity; the driver also rejects duplicate images.
    return digest([task, image_hash, question])


def read_image(root, name):
    from PIL import Image
    with Image.open(root / name) as im:
        return im.copy()


def wrap(task, tier, seed, split, index, record, image, audit=None, answer_type='text'):
    """A written record + its image -> Item.  The record and the audit stay in the in-memory meta, for the
    verifier; a choice question keeps its options as choices, a free-form one keeps any options it shows."""
    image = image.convert('RGB')
    question, gold = str(record['question']), answer(record)
    px = pixel_hash(image)
    meta = {'record': plain(record), 'audit': plain(audit or {}),
            'schedule': {'task': task, 'tier': tier, 'split': split, 'index': index, 'derived_seed': seed},
            'pixel_sha256': px}
    options = [str(o) for o in (record.get('options') or [])]
    is_choice = record.get('ansType') == 'choice'
    return api.Item(question=question, answer=gold, image=image, meta=meta, core_key=key(task, px, question),
                    answer_type='choice' if is_choice else answer_type,
                    choices=options if is_choice else None, options=None if is_choice else options)


def generator_tier(record):
    return record.get('tier', record.get('difficulty'))


def check_wrapped(task, record, image, tier_of=generator_tier):
    """The checks every wrapped record shares: schedule, question/answer, tier and rendered pixels.
    Returns (meta, generator record); raises ValueError on a mismatch."""
    meta = record['meta']
    schedule = meta['schedule']
    original = meta['record']
    if schedule['task'] != task:
        raise ValueError('wrong task in schedule')
    for field in ('task', 'tier', 'split'):
        if record[field] != schedule[field]:
            raise ValueError(f'{field} differs from the schedule')
    for field in ('index', 'derived_seed'):
        if record['source'][field] != schedule[field]:
            raise ValueError(f'{field} differs from the schedule')
    if record['question'] != original['question'] or record['answer'] != answer(original):
        raise ValueError('question/answer differs from the generator record')
    if tier_of(original) != record['tier']:
        raise ValueError('generator tier differs from the requested tier')
    px = pixel_hash(image)
    if image.size != (448, 448) or px != meta['pixel_sha256']:
        raise ValueError('image differs from the rendered pixels')
    if record['core_key'] != key(task, px, record['question']):
        raise ValueError('rendered question key mismatch')
    return meta, original


def run_checks(task, record, image, check, tier_of=generator_tier):
    """Run `check(record, meta, generator record, image)`; problems as a list."""
    try:
        meta, original = check_wrapped(task, record, image, tier_of)
        check(record, meta, original, image)
        return []
    except (KeyError, TypeError, ValueError, IndexError, AssertionError, OverflowError) as exc:
        return [f'{task}: {type(exc).__name__}: {exc}']


def build(task, builder, tier, seed, *, split, index, answer_type='text'):
    if tier not in api.TIERS or split not in api.SPLITS or index < 0:
        raise ValueError('invalid tier, split or candidate index')
    record, image, audit = builder(tier, seed, split, index)
    return wrap(task, tier, seed, split, index, record, image, audit, answer_type=answer_type)
