"""Procedural glyph scenes, with question-independent material/geometry checks.

Each candidate is a deterministic
function of seed/split/index/tier plus its externally scheduled conditions.
A rejected candidate keeps its identities, partition, query and typography fixed
through the bounded position search. The caller may retry a new candidate index.
"""

from __future__ import annotations
from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text

from functools import lru_cache
import hashlib
import itertools
import json
import math
import random

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import scipy

from . import glyph_render, partition_policy, query_policy, size_policy
from .base import materials, stroke_masks
from .base.materials import ID_MATERIALS, OOD_MATERIALS

VERSION = 'overlapping_glyphs_core'
# fixed seed namespace of every item (changing it changes every item)
_SEED_NAMESPACE = seed_text(769285801812975708478485294502306795297954091832859754735499564169637424)
SIZE = 448
VIEW = 368
VIEW_SCALE = VIEW / SIZE
MIN_FINAL_CORE = 0.38
MIN_BEFORE_OUTLINE = 0.40
MAX_LAYOUT_ATTEMPTS = 180
MAX_OUTLINE_ATTEMPTS = 140
ROTATION = dict(easy=18.0, medium=28.0, hard=40.0, xhard=60.0, xxhard=80.0)
CLEAN_OCCLUSION_CAP = dict(easy=0.30, medium=0.50, hard=0.55, xhard=0.62, xxhard=0.65)
OVERLAP_FLOOR = dict(easy=0.015, medium=0.10, hard=0.14, xhard=0.15, xxhard=0.15)
BACKGROUNDS = ('paper', 'blobs', 'bands', 'dots', 'grid', 'waves')
FRAME_SIDES = ('top', 'bottom', 'left', 'right')
PALETTE = {
    'yellow': (245, 188, 30),
    'green': (40, 158, 64),
    'blue': (40, 110, 210),
    'red': (224, 64, 50),
    'orange': (240, 140, 35),
    'purple': (150, 70, 190),
    'cyan': (25, 186, 205),
    'pink': (231, 121, 181),
    'brown': (112, 66, 32),
    'black': (27, 29, 32),
}
EASY_ALPHABET = 'ABEFGHKMNPRTWXYZ234679'
HARD_ALPHABET = ''.join(sorted(set(EASY_ALPHABET + 'OQ0CGDS51IL8')))


class GenerationRejected(RuntimeError):
    """A bounded, otherwise valid candidate failed its layout/visibility checks."""

    def __init__(self, reason, *, seed, index, tier, split, rejection_counts, groups=None):
        super().__init__(f'{split}/{tier}/{index}: {reason}')
        self.reason = reason
        self.metadata = dict(
            seed=seed,
            index=index,
            tier=tier,
            split=split,
            reason=reason,
            rejection_counts=dict(rejection_counts),
            groups=groups,
        )
        self.rejection_counts = dict(rejection_counts)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def derive_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8], 'big')


@lru_cache(maxsize=1)
def source_fingerprint():
    here = _ROOT / 'tasks/discrimination_grouping/overlapping_glyphs/generator'
    paths = [
        here / name
        for name in ('glyph_core.py', 'glyph_render.py', 'partition_policy.py', 'query_policy.py', 'size_policy.py')
    ]
    paths += [here / 'base' / name for name in ('stroke_masks.py', 'materials.py', 'rope_renderer.py')]
    hashes = {str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashes, hashlib.sha256(canonical(hashes).encode()).hexdigest()


def _put(local, center, mode='RGBA'):
    full = Image.new(mode, (SIZE, SIZE))
    xy = (round(center[0] - local.width / 2), round(center[1] - local.height / 2))
    if mode == 'RGBA':
        full.alpha_composite(local, xy)
    else:
        full.paste(local, xy)
    return full


def _resize_alpha(alpha):
    alpha = np.asarray(alpha)
    if np.issubdtype(alpha.dtype, np.floating):
        alpha = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(alpha).resize((VIEW, VIEW), Image.Resampling.LANCZOS))


def _components(masks):
    n = len(masks)
    adjacent = np.zeros((n, n), bool)
    sizes = [int(m.sum()) for m in masks]
    for i in range(n):
        for j in range(i):
            adjacent[i, j] = adjacent[j, i] = np.count_nonzero(masks[i] & masks[j]) >= max(
                24, 0.008 * min(sizes[i], sizes[j])
            )
    remaining = set(range(n))
    components = []
    while remaining:
        seen = {min(remaining)}
        todo = list(seen)
        while todo:
            new = set(map(int, np.flatnonzero(adjacent[todo.pop()]))) - seen
            seen.update(new)
            todo.extend(new)
        remaining.difference_update(seen)
        components.append(sorted(seen))
    return components, adjacent


def _winner(masks, beats):
    n = len(masks)
    coverage = np.zeros(masks[0].shape, np.uint16)
    for i, mask in enumerate(masks):
        coverage |= mask.astype(np.uint16) << i
    lut = np.full(1 << n, -1, np.int8)
    for code in range(1, 1 << n):
        members = [i for i in range(n) if code >> i & 1]
        lut[code] = max(members, key=lambda i: sum(beats[i][j] for j in members if j != i))
    return lut[coverage]


def _draw_order(n, groups, mode, rng):
    if mode == 'global':
        order = rng.sample(range(n), n)
        ranks = {i: r for r, i in enumerate(order)}
        return [[ranks[i] > ranks[j] for j in range(n)] for i in range(n)], order
    # A local tournament may have no visible 3-cycle when only pairs intersect.
    for _ in range(100):
        beats = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                beats[i][j] = bool(rng.getrandbits(1))
                beats[j][i] = not beats[i][j]
        triples = [t for ids in groups for t in itertools.combinations(ids, 3)]
        if not triples or any(
            (beats[a][b] and beats[b][c] and beats[c][a]) or (beats[a][c] and beats[c][b] and beats[b][a])
            for a, b, c in triples
        ):
            return beats, None
    raise RuntimeError('Could not draw local tournament')


def _materials(tier, groups, rng):
    if tier != 'xxhard':
        chosen = [rng.choice(ID_MATERIALS) for _ in groups]
        return [chosen[g] for g, k in enumerate(groups) for _ in range(k)], 'id_per_cluster'
    family = rng.choice(('known_per_glyph', 'novel_per_cluster', 'all_per_glyph'))
    if family == 'novel_per_cluster':
        chosen = [rng.choice(OOD_MATERIALS) for _ in groups]
        return [chosen[g] for g, k in enumerate(groups) for _ in range(k)], family
    pool = ID_MATERIALS if family == 'known_per_glyph' else ID_MATERIALS + OOD_MATERIALS
    return [rng.choice(pool) for _ in range(sum(groups))], family


def _prepare_entries(tier, groups, ns):
    n = sum(groups)
    chars = random.Random(ns('identities')).sample(EASY_ALPHABET if tier in ('easy', 'medium') else HARD_ALPHABET, n)
    colors = random.Random(ns('colors')).sample(list(PALETTE), n)
    material_names, family = _materials(tier, groups, random.Random(ns('materials')))
    count = len(groups)
    base = {'easy': 166.0, 'medium': 180.0, 'hard': 194.0, 'xhard': 202.0, 'xxhard': 205.0}[tier]
    if count >= 3:
        base = 175.0
    elif count == 2:
        base = 188.0 if max(groups) < 6 else 175.0
    if tier == 'easy':
        rng = random.Random(ns('font_sizes'))
        sizes = [round(base * rng.uniform(0.94, 1.06)) for _ in range(n)]
        size_meta = dict(policy='easy_small_variation', base_font_px=base)
    else:
        sizes, size_meta = size_policy.sample_font_sizes(base, n, ns('font_sizes'), preview_contrast=None)
    angle_rng = random.Random(ns('rotation'))
    width_rng = random.Random(ns('stroke_width'))
    entries = []
    group_index = [g for g, k in enumerate(groups) for _ in range(k)]
    for i in range(n):
        angle = angle_rng.uniform(-ROTATION[tier], ROTATION[tier])
        width = width_rng.uniform(0.72, 1.34)
        mask, width_meta = stroke_masks.make_stroke_mask(chars[i], sizes[i], width)
        material_seed = ns('material_texture', i)
        layer = materials.render_material(mask, PALETTE[colors[i]], material_names[i], material_seed)
        clean = mask.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        support = np.asarray(layer)[..., 3] >= 48
        ys, xs = np.nonzero(support)
        if not len(xs):
            raise ValueError('Empty rendered material')
        bounds = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        entries.append(
            dict(
                id=i,
                char=chars[i],
                color=colors[i],
                group=group_index[i],
                font_px=int(sizes[i]),
                angle=float(angle),
                width=width_meta,
                material=material_names[i],
                material_seed=material_seed,
                clean=clean,
                layer=layer,
                local_support_bounds=bounds,
                local_support_area=int(support.sum()),
                clean_area=int((np.asarray(clean) > 127).sum()),
            )
        )
    return entries, family, size_meta


def _positions(entries, groups, rng):
    count = len(groups)
    slots = {
        1: [(224, 224)],
        2: [(124, 124), (324, 324)],
        3: [(110, 109), (337, 109), (224, 334)],
        4: [(110, 110), (338, 110), (110, 338), (338, 338)],
    }[count]
    slots = rng.sample(slots, len(slots))
    poses = [None] * len(entries)
    for gi, ids in enumerate(groups):
        cx, cy = slots[gi]
        cx += rng.uniform(-9, 9)
        cy += rng.uniform(-9, 9)
        k = len(ids)
        fp = float(np.mean([entries[i]['font_px'] for i in ids]))
        radius = fp * rng.uniform(0.22, 0.43) * math.sin(math.pi / 5) / math.sin(math.pi / k) if k > 1 else 0
        phase = rng.uniform(0, 2 * math.pi)
        for i, slot in zip(ids, rng.sample(range(k), k)):
            a = phase + 2 * math.pi * slot / k
            poses[i] = [
                round(cx + radius * math.cos(a) + rng.uniform(-4, 4)),
                round(cy + radius * math.sin(a) + rng.uniform(-4, 4)),
            ]
        boxes = []
        for i in ids:
            e = entries[i]
            x0, y0, x1, y1 = e['local_support_bounds']
            ox = round(poses[i][0] - e['layer'].width / 2)
            oy = round(poses[i][1] - e['layer'].height / 2)
            boxes.append((x0 + ox, y0 + oy, x1 + ox, y1 + oy))
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        if x1 - x0 > 428 or y1 - y0 > 428:
            return None
        dx = max(0, 9 - x0) - max(0, x1 - 438)
        dy = max(0, 9 - y0) - max(0, y1 - 438)
        for i in ids:
            poses[i][0] += dx
            poses[i][1] += dy
    return poses


def _material_state(entries, poses, beats, mode):
    layers = [_put(e['layer'], p) for e, p in zip(entries, poses)]
    arrays = [np.array(layer) for layer in layers]
    alpha = [a[..., 3].copy() for a in arrays]
    if mode == 'global':
        order = sorted(range(len(entries)), key=lambda i: sum(beats[i]))
    else:
        winner = _winner([a >= 128 for a in alpha], beats)
        for i, a in enumerate(arrays):
            a[..., 3][(winner >= 0) & (winner != i)] = 0
        order = list(range(len(entries)))
    foreground = Image.new('RGBA', (SIZE, SIZE))
    remaining = np.ones((SIZE, SIZE), np.float32)
    contribution = [None] * len(entries)
    for i in reversed(order):
        own = arrays[i][..., 3].astype(np.float32) / 255.0
        contribution[i] = own * remaining
        remaining *= 1 - own
    for i in order:
        foreground = Image.alpha_composite(foreground, Image.fromarray(arrays[i]))
    clean = [_resize_alpha(np.asarray(_put(e['clean'], p, 'L'))) > 127 for e, p in zip(entries, poses)]
    isolated = [(_resize_alpha(a) >= 128) & core for a, core in zip(alpha, clean)]
    visible = [(_resize_alpha(a) >= 128) & core & iso for a, core, iso in zip(contribution, clean, isolated)]
    return foreground, alpha, clean, isolated, contribution, visible


def _fit(entries, wanted, beats, mode, tier, rng, rejected):
    for attempt in range(MAX_LAYOUT_ATTEMPTS):
        poses = _positions(entries, wanted, rng)
        if poses is None:
            rejected['bounds'] += 1
            continue
        masks = [np.asarray(_put(e['clean'], p, 'L')) > 127 for e, p in zip(entries, poses)]
        if any(int(m.sum()) != e['clean_area'] for m, e in zip(masks, entries)):
            rejected['clean_clipping'] += 1
            continue
        components, adj = _components(masks)
        if sorted(components) != sorted(wanted):
            rejected['components'] += 1
            continue
        alphas = [np.asarray(_put(e['layer'], p))[..., 3] for e, p in zip(entries, poses)]
        supports = [a >= 48 for a in alphas]
        if any(int(a.sum()) != e['local_support_area'] for a, e in zip(supports, entries)):
            rejected['material_clipping'] += 1
            continue
        group_supports = [np.logical_or.reduce([supports[i] for i in ids]) for ids in wanted]
        if any(np.any(group_supports[i] & group_supports[j]) for i in range(len(wanted)) for j in range(i)):
            rejected['cross_group'] += 1
            continue
        final_supports = [_resize_alpha(a) >= 48 for a in alphas]
        final_groups = [np.logical_or.reduce([final_supports[i] for i in ids]) for ids in wanted]
        if any(np.any(final_groups[i] & final_groups[j]) for i in range(len(wanted)) for j in range(i)):
            rejected['cross_group'] += 1
            continue
        winner = _winner(masks, beats)
        occ = [1 - int((m & (winner == i)).sum()) / max(1, int(m.sum())) for i, m in enumerate(masks)]
        if max(occ) > CLEAN_OCCLUSION_CAP[tier]:
            rejected['clean_occlusion'] += 1
            continue
        overlapping = [i for ids in wanted if len(ids) > 1 for i in ids]
        if overlapping and np.mean([occ[i] for i in overlapping]) < OVERLAP_FLOOR[tier]:
            rejected['insufficient_overlap'] += 1
            continue
        state = _material_state(entries, poses, beats, mode)
        visible_fraction = [int(v.sum()) / max(1, int(c.sum())) for v, c in zip(state[-1], state[2])]
        if min(visible_fraction) < MIN_BEFORE_OUTLINE:
            rejected['material_visibility'] += 1
            continue
        return poses, occ, state, attempt + 1, adj
    return None


def _visible_with_outlines(state, shapes):
    canonical_mask = glyph_render.raster_outlines(shapes)
    final_mask = _resize_alpha((canonical_mask * 255).astype(np.uint8)) >= 128
    clean, isolated, contribution = state[2:5]
    visible = [
        (_resize_alpha(a * (~canonical_mask)) >= 128) & c & iso & ~final_mask
        for a, c, iso in zip(contribution, clean, isolated)
    ]
    return visible


def _outlines(entries, poses, state, rng, rejected):
    clean, visible = state[2], state[-1]
    wanted = rng.choice((1, 2))
    shapes = []
    for _ in range(wanted):
        previous = _visible_with_outlines(state, shapes)
        for attempt in range(MAX_OUTLINE_ATTEMPTS):
            proposal = glyph_render.sample_outline(rng, poses[rng.randrange(len(entries))])
            candidate = shapes + [proposal]
            after = _visible_with_outlines(state, candidate)
            losses = [int((v & ~a).sum()) / max(1, int(c.sum())) for v, a, c in zip(visible, after, clean)]
            remains = [int(a.sum()) / max(1, int(c.sum())) for a, c in zip(after, clean)]
            newly = sum(int((old & ~a).sum()) for old, a in zip(previous, after))
            if max(losses) <= 0.10 and min(remains) >= MIN_FINAL_CORE and newly >= 12:
                shapes.append(proposal)
                break
            rejected['outline_placement'] += 1
        else:
            return None
    return shapes


def _visibility_metadata(entries, poses, occ, state, shapes):
    foreground, alphas, clean, isolated, contribution, visible = state
    final_visible = _visible_with_outlines(state, shapes)
    reports = []
    for i, (e, p, c, iso, v, a) in enumerate(zip(entries, poses, clean, isolated, visible, alphas)):
        final = final_visible[i]
        area = int(c.sum())
        available = int(iso.sum())
        seen = int(final.sum())
        support = _resize_alpha(a) >= 48
        ys, xs = np.nonzero(support)
        reports.append(
            dict(
                id=i,
                char=e['char'],
                color=e['color'],
                material=e['material'],
                group=e['group'],
                core_area_px=area,
                isolated_material_core_px=available,
                actual_visible_core_px=seen,
                actual_core_visible_fraction=seen / area,
                actual_core_missing_fraction=1 - seen / area,
                intrinsic_material_core_missing_fraction=1 - available / area,
                extra_occlusion_fraction_of_core=(available - seen) / area,
                logical_clean_occlusion=occ[i],
                outline_extra_hidden_core_px=int((v & ~final).sum()),
                final_material_bounds_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                clipped_material_support_px=e['local_support_area'] - int((a >= 48).sum()),
            )
        )
    return dict(
        version='material_core_visibility',
        glyphs=reports,
        definition='alpha contribution >=128 inside clean glyph core at final368px, excluding outline strokes',
        core_alpha_threshold=128,
        support_alpha_threshold=48,
        final_local_resolution=[VIEW, VIEW],
        minimum_final_core_visible=min(g['actual_core_visible_fraction'] for g in reports),
        any_clipped_material_support=any(g['clipped_material_support_px'] for g in reports),
        any_cross_group_material_overlap=False,
        limitations=[
            'material holes are separately reported, not equivalent to other-object occlusion',
            'not a full alternative-glyph readability certificate or a human accuracy guarantee',
        ],
    )


def make_sample(
    seed: int,
    index: int,
    tier: str,
    mode: str,
    outline: bool,
    split: str,
    *,
    subset: bool | None = None,
    groups: list[int] | None = None,
):
    """Return (RGB448 image, metadata), or raise GenerationRejected.

    The runner schedules mode/outline/subset. None samples a 50% proper-subset
    question independently of image content. All scene glyphs remain rendered.
    """
    for value, name in ((seed, 'seed'), (index, 'index')):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} must be int')
    if index < 0:
        raise ValueError('index must be nonnegative')
    if tier not in ROTATION or split not in ('train', 'eval', 'validation', 'ood'):
        raise ValueError('unknown tier or split')
    if (tier == 'xxhard') != (split == 'ood'):
        raise ValueError('xxhard is reserved exclusively for ood')
    if mode not in ('global', 'cyclic') or not isinstance(outline, bool):
        raise ValueError('invalid compositor/outline')
    if subset is not None and not isinstance(subset, bool):
        raise ValueError('subset must be bool or None')
    ns = lambda name, *extra: derive_seed(_SEED_NAMESPACE, seed, split, index, tier, name, *extra)
    groups = partition_policy.sample_partition(tier, ns('partition')) if groups is None else list(groups)
    partition_policy.validate_partition(tier, groups)
    n = sum(groups)
    cursor = 0
    wanted = []
    for k in groups:
        wanted.append(list(range(cursor, cursor + k)))
        cursor += k
    query_ids = query_policy.select_query_ids(tier, n, ns('query'), subset=subset)
    entries, family, size_meta = _prepare_entries(tier, groups, ns)
    beats, order = _draw_order(n, wanted, mode, random.Random(ns('depth_order')))
    rejected = {
        k: 0
        for k in (
            'bounds',
            'clean_clipping',
            'components',
            'material_clipping',
            'cross_group',
            'clean_occlusion',
            'insufficient_overlap',
            'material_visibility',
            'outline_placement',
        )
    }
    fitted = _fit(entries, wanted, beats, mode, tier, random.Random(ns('placement')), rejected)
    if fitted is None:
        raise GenerationRejected(
            'bounded_layout_search_exhausted',
            seed=seed,
            index=index,
            tier=tier,
            split=split,
            rejection_counts=rejected,
            groups=groups,
        )
    poses, occ, state, attempts, adj = fitted
    shapes = _outlines(entries, poses, state, random.Random(ns('outlines')), rejected) if outline else []
    if shapes is None:
        raise GenerationRejected(
            'bounded_outline_search_exhausted',
            seed=seed,
            index=index,
            tier=tier,
            split=split,
            rejection_counts=rejected,
            groups=groups,
        )
    visibility = _visibility_metadata(entries, poses, occ, state, shapes)
    if visibility['minimum_final_core_visible'] < MIN_FINAL_CORE:
        raise RuntimeError('internal final visibility invariant failed')
    bg_rng = random.Random(ns('background'))
    background = bg_rng.choice(BACKGROUNDS)
    image, bg_meta = glyph_render.procedural_background(ns('background_texture'), background)
    frame_rng = random.Random(ns('frames'))
    side = frame_rng.choice(FRAME_SIDES)
    viewport = {'bottom': (40, 4), 'top': (40, 76), 'left': (76, 40), 'right': (4, 40)}[side]
    fg = glyph_render.draw_outlines(state[0], shapes).resize((VIEW, VIEW), Image.Resampling.LANCZOS)
    image = image.convert('RGBA')
    image.alpha_composite(fg, viewport)
    image = image.convert('RGB')
    colors = [entries[i]['color'] for i in query_ids]
    boxes, strip = glyph_render.draw_query_frames(image, colors, side, ns('frame_offset'), PALETTE)
    direction = 'left to right' if side in ('top', 'bottom') else 'top to bottom'
    question = (
        f'The image contains {n} uppercase letters or Arabic numerals in different colors. '
        f'Read the characters matching the colored boxes on the {side}, '
        f'in box order from {direction}. Separate the characters with commas.'
    )
    answer = ','.join(entries[i]['char'] for i in query_ids)
    glyphs = []
    for e, p in zip(entries, poses):
        glyphs.append(
            {
                k: v
                for k, v in e.items()
                if k not in ('clean', 'layer', 'local_support_bounds', 'local_support_area', 'clean_area')
            }
        )
        glyphs[-1].update(pose=list(map(int, p)), font_px_final=e['font_px'] * VIEW_SCALE)
    hashes, source_hash = source_fingerprint()
    meta = dict(
        generator_version=VERSION,
        seed=seed,
        index=index,
        candidate_index=index,
        split=split,
        tier=tier,
        scene_seed=ns('scene'),
        rng_namespaces='SHA256(version,seed,split,index,tier,branch,optional-glyph-id)',
        groups=groups,
        components=wanted,
        adjacency=adj.astype(int).tolist(),
        glyphs=glyphs,
        mode=mode,
        beats=beats,
        global_order=order,
        outline=outline,
        neutral_shapes=shapes,
        material_family=family,
        ood_family=family if split == 'ood' else None,
        font_size_sampling=size_meta,
        rotation_limit_deg=ROTATION[tier],
        stroke_scale_range=[0.72, 1.34],
        query_ids=query_ids,
        query_colors=colors,
        query_seed=ns('query'),
        query_policy=query_policy.POLICY,
        swatch_order=colors,
        swatch_boxes=boxes,
        frame_strip=strip,
        frame_side=side,
        frame_direction=direction,
        viewport_xy=list(viewport),
        query_display='queried_colors_only_in_answer_order',
        question=question,
        answer=answer,
        n_characters=n,
        n_queried=len(query_ids),
        proper_subset=len(query_ids) < n,
        background=background,
        background_metadata=bg_meta,
        final_material_visibility=visibility,
        layout_attempts=attempts,
        rejection_counts=rejected,
        current_full_rival_certificate=False,
        source_hashes=hashes,
        generator_source_sha256=source_hash,
        font_sha256=stroke_masks.FONT_SHA256,
        materials_version=materials.LABEL,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        pillow_version=PIL_VERSION,
        image_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        source_profile='source_rgb448_square',
    )
    meta['scene_sha256'] = hashlib.sha256(
        canonical(
            dict(
                glyphs=glyphs,
                groups=groups,
                beats=beats,
                mode=mode,
                shapes=shapes,
                background=bg_meta,
                viewport=viewport,
            )
        ).encode()
    ).hexdigest()
    return image, meta
