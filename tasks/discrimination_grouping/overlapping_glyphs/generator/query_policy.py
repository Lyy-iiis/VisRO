"""Final-answer query policy: half proper subsets, half full readouts."""
import math
import random

POLICY = {
    'version': 'glyph_query', 'proper_subset_probability': .5,
    'easy': '1..n-1 if subset', 'medium': '2..n-1 if subset',
    'hard': '3..n-1 if subset', 'xhard': '3..n-1 if subset',
    'xxhard': 'ceil(n/2)..n-1 if subset', 'max_answers': 8,
    'selection': 'choose subset/full first; uniform subset count, subset and ordering',
    'image': 'all glyphs remain; only queried colors have frames in answer order',
    'reading_order': 'horizontal left to right; vertical top to bottom',
}

def query_count_range(tier, n):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError('expected 1..8 scene glyphs')
    lower = {'easy':1, 'medium':2, 'hard':3, 'xhard':3, 'xxhard':math.ceil(n/2)}[tier]
    if lower >= n:
        raise ValueError('scene does not permit both subset and full queries')
    return lower, n

def select_query_ids(tier, n, seed, *, subset=None):
    lower, upper = query_count_range(tier, n)
    if subset is not None and not isinstance(subset, bool):
        raise ValueError('subset must be bool or None')
    rng = random.Random(seed)
    use_subset = rng.random() < POLICY['proper_subset_probability'] if subset is None else subset
    count = rng.randint(lower, upper-1) if use_subset else upper
    return rng.sample(range(n), count)
