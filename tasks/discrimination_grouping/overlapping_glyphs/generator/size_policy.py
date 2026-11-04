"""Per-glyph font sizes, sampled without question or color information."""
import random

POLICY = {
    'version': 'per_glyph_font_size',
    'starts_at': 'medium',
    'per_glyph_multiplier': [.78, 1.28],
    'scene_multiplier': [.95, 1.05],
    'minimum_final_font_px': 110,
    'query_independent': True,
    'preview_seed_selection': 'first draw with at least 1.35 max/min font contrast',
}

def sample_font_sizes(base_px, count, seed, preview_contrast=None):
    rng = random.Random(seed)
    scene_scale = rng.uniform(*POLICY['scene_multiplier'])
    minimum = round(POLICY['minimum_final_font_px'] / (368/448))
    for attempt in range(100):
        multipliers = [rng.uniform(*POLICY['per_glyph_multiplier']) for _ in range(count)]
        sizes = [max(minimum, round(base_px * scene_scale * value)) for value in multipliers]
        if preview_contrast is None or max(sizes) / min(sizes) >= preview_contrast:
            return sizes, dict(seed=seed, base_font_px=base_px, scene_multiplier=scene_scale,
                sampled_multipliers=multipliers, preview_draw=attempt+1,
                actual_min_max_ratio=max(sizes)/min(sizes),
                contrast_filter=preview_contrast)
    raise RuntimeError('Could not sample visible font-size variation')
