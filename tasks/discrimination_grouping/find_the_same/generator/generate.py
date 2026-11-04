"""Base generator of find_the_same, in the style of BabyVision's "Find the same" task.

Renders M numbered items (1..M) drawn from a shared library of ~16k REAL icons
(Material Design Icons + game-icons.net silhouettes, OpenMoji color emoji). Some
items are IDENTICAL in pairs (the SAME rendered tile pasted at both slots -> byte
-for-byte identical bitmap); the rest are unique distractors. The task is to find
ALL identical index pairs.

Answer format: "1-7,2-9,3-10"
The stored string is canonical (a<b, pairs sorted); grading ignores both orders.

Representation level: **mid** (shape abstraction & correspondence).
The bottleneck is grouping each item into its parts and matching the WHOLE icon
across the grid, against distractors that include subtle "near-twins" -- a copy
of a real pair's icon with ONE small alteration (a part removed/added, a rotation
of at least 12 degrees, a mirror, a recolor/hue-shift; never a pure shift or
scale) -- so you must inspect every part carefully. You compare real shapes, not
read text.

Icon diversity
--------------
Icons come from `iconlib` (~16k real icons) instead of a single hand-coded
critter family, so each problem shows a genuinely different object and the set is
not memorizable. Two styles:
  * silhouette  (MDI + game-icons, tinted ONE color for the whole grid)
  * color       (OpenMoji, full color, multi-part)
For silhouette grids one tint is shared across ALL items, so "same" means same
SHAPE -- you cannot shortcut the match by color.

Correctness and final-resolution readability
-------------------------------------------
True pairs use the same bitmap, and gold is computed from tiles composited on
the actual question background. Every nonmatching pair must have a connected
patch of >=48 pixels with a channel difference >=64, including >=12 interior
pixels after 3x3 erosion, both as placed and after the best small translation
of one tile onto the other (so a mirror of a symmetric icon that is merely
moved a few pixels is rejected). A near-twin changes at most 55% of its pair's
ink pixels and is strictly closer to its own pair than to any other pair.
Invisible alpha differences, weak antialiasing, and thin one-pixel edges alone
are insufficient. These are conservative screening rules, not a measured
guarantee of human readability.

The answer is a set of unordered pairs. Missing/extra pairs are wrong;
reordering pairs or reversing their members never changes the score.

DIFFICULTY KNOB (the shape-similarity axis for this mid-level task)
------------------------------------------------------------------
`difficulty` controls:
  * M            : number of items in the grid (more clutter to scan).
  * n_pairs      : how many identical pairs are hidden.
  * near-twins   : how many distractors are subtle alterations of a real pair
                   (the harder the level, the more near-twins and the subtler the
                   alteration strength), forcing real part-by-part inspection.

Each problem is a pure function of (seed, idx): random.Random(f"{seed}:{idx}").
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from shared.drawing.fonts import load_bundled_font
import shared.extra_difficulty as extra

GENERATOR_VERSION = "readable-pairs"
MIN_CONTRAST = 64
MIN_COMPONENT_PIXELS = 48
MIN_COMPONENT_CORE = 12
# Two tiles must also differ after the best small
# translation (a mirrored symmetric icon whose ink is off-centre in its viewBox
# is otherwise the same picture moved a few px), and a near-twin must stay near
# its own pair: at most TWIN_MAX_CHANGE of the pair's ink pixels changed on the
# delivered tile, and strictly closer to its own pair than to any other pair.
POSE_SEARCH = 2          # px searched around the ink-bbox offset (and bbox size slack)
INK_LEVEL = 32           # max-channel difference from BG that counts as ink
TWIN_MAX_CHANGE = (55, 100)   # changed px / base ink px <= 55/100

BG = (252, 250, 246)
SS = 4  # supersampling for crisp tiles

# dark-ish tints for silhouette grids (whole grid shares ONE)
SIL_COLORS = [
    (30, 30, 30), (25, 60, 120), (120, 35, 35), (30, 95, 55),
    (70, 45, 105), (140, 80, 20), (20, 90, 100),
]


def derive_rng(seed: int, idx: int) -> random.Random:
    return random.Random(f"{seed}:{idx}")


def displayed_tile(tile):
    """Composite exactly as the production question, on its actual background."""
    image = Image.new("RGB", tile.size, BG)
    image.paste(tile, (0, 0), tile if tile.mode == "RGBA" else None)
    return image


def displayed_array(tile):
    return np.asarray(displayed_tile(tile)).astype(np.int16)


def array_metrics(aa, bb):
    """difference_metrics on two displayed int16 RGB arrays."""
    difference = np.max(np.abs(aa - bb), axis=2)
    mask = difference >= MIN_CONTRAST
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    core = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    cores = np.bincount(labels[core], minlength=count + 1)
    eligible = [i for i in range(1, count + 1)
                if areas[i] >= MIN_COMPONENT_PIXELS and cores[i] >= MIN_COMPONENT_CORE]
    best = max(eligible, key=lambda i: areas[i]) if eligible else (int(np.argmax(areas[1:])) + 1 if count else 0)
    return {"readable": bool(eligible), "max_channel_difference": int(difference.max()),
            "contrast_pixels": int(mask.sum()),
            "component_pixels": int(areas[best]) if best else 0,
            "component_core_pixels": int(cores[best]) if best else 0}


_BG16 = np.array(BG, np.int16)


def ink_box(arr):
    """(y0, x0, y1, x1) of the pixels at least INK_LEVEL from the background."""
    ys, xs = np.nonzero(np.abs(arr - _BG16).max(axis=2) >= INK_LEVEL)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def translate(arr, dy, dx):
    """arr moved by (dy, dx) px; uncovered pixels become the background."""
    out = np.empty_like(arr)
    out[:] = _BG16
    H, W = arr.shape[:2]
    if abs(dy) >= H or abs(dx) >= W:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        arr[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def best_alignments(aa, bb):
    """Translations of bb onto aa with the fewest high-contrast pixels.

    Only tiles whose ink boxes agree in size within POSE_SEARCH px can be the
    same picture moved; for those, every shift within POSE_SEARCH of the ink-box
    offset, plus (0, 0), is tried.  Returns (count, [shifts]) or None.
    """
    ba, bb_ = ink_box(aa), ink_box(bb)
    if ba is None or bb_ is None:
        return None
    if abs((ba[2] - ba[0]) - (bb_[2] - bb_[0])) > POSE_SEARCH or \
            abs((ba[3] - ba[1]) - (bb_[3] - bb_[1])) > POSE_SEARCH:
        return None
    oy, ox = ba[0] - bb_[0], ba[1] - bb_[1]
    shifts = {(0, 0)} | {(oy + i, ox + j) for i in range(-POSE_SEARCH, POSE_SEARCH + 1)
                         for j in range(-POSE_SEARCH, POSE_SEARCH + 1)}
    counts = {sh: int((np.abs(aa - translate(bb, *sh)).max(axis=2) >= MIN_CONTRAST).sum())
              for sh in shifts}
    best = min(counts.values())
    return best, sorted(sh for sh, c in counts.items() if c == best)


def pose_distinct(aa, bb):
    """The floor must still hold after the best small translation, both ways.

    Every best-aligned shift (ties included) must leave a readable patch, so a
    shift-only copy -- e.g. a mirror of a symmetric icon that sits off-centre in
    its viewBox -- is never accepted as a different item.
    """
    for x, y in ((aa, bb), (bb, aa)):
        got = best_alignments(x, y)
        if got is not None and not all(array_metrics(x, translate(y, *sh))["readable"] for sh in got[1]):
            return False
    return True


def distinct_arrays(aa, bb):
    return array_metrics(aa, bb)["readable"] and pose_distinct(aa, bb)


def change_counts(cand, base):
    """(pixels changed by >= MIN_CONTRAST, ink pixels of base) on displayed arrays."""
    changed = int((np.abs(cand - base).max(axis=2) >= MIN_CONTRAST).sum())
    ink = int((np.abs(base - _BG16).max(axis=2) >= INK_LEVEL).sum())
    return changed, ink


def twin_is_near(cand, base_arr, other_pair_arrs):
    """Near-twin gate: <= TWIN_MAX_CHANGE of the base's ink changed, and the
    base strictly the nearest pair by that fraction (compared as integers)."""
    c, k = change_counts(cand, base_arr)
    if c * TWIN_MAX_CHANGE[1] > TWIN_MAX_CHANGE[0] * k:
        return False, (c, k)
    for other in other_pair_arrs:
        c2, k2 = change_counts(cand, other)
        if not c * k2 < c2 * k:
            return False, (c, k)
    return True, (c, k)


# --------------------------------------------------------------------------- difficulty
def diff_params(diff, rng):
    """Return (M, n_pairs, n_twin, strength). n_twin = near-twin decoys."""
    if diff == "easy":
        M = rng.choice([9, 10, 11])
        n_pairs = rng.randint(2, 3)
        n_twin = 0
        strength = rng.uniform(0.7, 1.0)      # alteration obvious (n/a if no twins)
    elif diff == "medium":
        M = rng.choice([11, 12, 14])
        n_pairs = rng.randint(3, 4)
        n_twin = rng.randint(1, 2)
        strength = rng.uniform(0.4, 0.7)
    elif diff == "hard":
        M = rng.choice([14, 15, 16])
        n_pairs = rng.randint(4, 5)
        n_twin = rng.randint(2, 3)
        strength = rng.uniform(0.18, 0.4)
    elif diff == "xhard":
        M, n_pairs, n_twin = 20, 6, 6
        strength = rng.uniform(0.30, 0.50)
    elif diff == "xxhard":
        M, n_pairs, n_twin = 25, 8, 8
        strength = rng.uniform(0.25, 0.45)
    else:
        raise ValueError(diff)
    return M, n_pairs, n_twin, strength


# --------------------------------------------------------------------------- build set
def build_items(lib, rng, M, n_pairs, n_twin, strength, tile, style, color):
    """Build the list of item tiles + ground-truth groups.

    Returns (tiles, groups, metas) or None to resample.
      tiles[i]  : RGBA PIL tile (tile x tile)
      groups[i] : pair-group id (>=0 shared within a true pair) or -1 (singleton)
      metas[i]  : per-item dict {icon_id, kind, alteration}

    Construction:
      * Pick distinct icon ids whose final-resolution rasters are also distinct.
      * Each PAIR renders its icon ONCE -> the SAME tile object is placed twice
        (byte-identical guaranteed).
      * Singletons: `n_twin` of them are NEAR-TWINS (a pair's icon + one
        lib.alter), the rest are distinct fresh icons.
      * Final tile comparisons prevent SVG aliases from creating accidental
        equality; near-twins are also verified against EVERY existing tile.
    """
    n_singles = M - 2 * n_pairs
    if n_singles < 0:
        return None
    n_twin = min(n_twin, n_singles, n_pairs)
    n_fresh = n_singles - n_twin

    S = tile * SS
    area = tile * tile

    def good_tile(t):
        """Reject degenerate renders that wreck uniqueness.

        Some silhouette SVGs are full-bleed (a solid background rect): tinted to
        the shared color they collapse to identical solid squares. Near-empty
        tiles are also useless. Keep tiles whose foreground footprint is a sane
        fraction of the tile.
        """
        fg = int((np.max(np.abs(np.asarray(displayed_tile(t)).astype(int) - BG), axis=2) >= 32).sum())
        return 0.03 * area <= fg <= 0.92 * area

    used_ids = set()
    tiles, groups, metas = [], [], []
    arrs = []                             # displayed int16 arrays, parallel to tiles

    def fresh_icon():
        """Pick an icon whose final tile differs from every existing tile.

        Distinct source IDs are insufficient because aliases and near-duplicate
        SVGs can rasterize to the same pixels.
        """
        for _ in range(300):
            e = lib.pick(rng, style=style)
            if e["id"] in used_ids:
                continue
            full = lib.render(e, S, color=color)
            t = full.resize((tile, tile), Image.LANCZOS)
            if not good_tile(t):
                continue
            ta = displayed_array(t)
            if not all(distinct_arrays(ta, existing) for existing in arrs):
                continue
            used_ids.add(e["id"])
            return e, full, t
        return None

    # --- pairs: render once, place twice (byte-identical) ---
    pair_entries = []
    pair_fulls = []
    for gi in range(n_pairs):
        got = fresh_icon()
        if got is None:
            return None
        e, full, t = got
        pair_entries.append(e)
        pair_fulls.append(full)
        for _ in range(2):
            tiles.append(t)               # SAME object -> identical bytes
            arrs.append(displayed_array(t))
            groups.append(gi)
            metas.append({"icon_id": e["id"], "kind": "pair", "alteration": None,
                          "twin_change": None})
    pair_arrs = [arrs[2 * gi] for gi in range(n_pairs)]

    def differs_from_all(cand_arr):
        return all(distinct_arrays(cand_arr, a) for a in arrs)

    # --- near-twin decoys: a pair's icon, altered subtly, must differ from all ---
    n_twin_done = 0
    for _ in range(n_twin * 40):
        if n_twin_done >= n_twin:
            break
        gi = rng.randrange(n_pairs)
        base_full = pair_fulls[gi]
        entry = pair_entries[gi]
        # try escalating strength until the change survives downsample & is unique
        placed = False
        for boost in (1.0, 1.5, 2.2):
            s_eff = min(1.0, strength * boost)
            alt_full, method, desc = lib.alter(entry, base_full, rng, s_eff,
                                               color=color)
            alt_tile = alt_full.resize((tile, tile), Image.LANCZOS)
            if not good_tile(alt_tile):
                continue
            alt_arr = displayed_array(alt_tile)
            near, change = twin_is_near(alt_arr, pair_arrs[gi],
                                        [a for j, a in enumerate(pair_arrs) if j != gi])
            if not near or not differs_from_all(alt_arr):
                continue
            tiles.append(alt_tile)
            arrs.append(alt_arr)
            groups.append(-1)
            metas.append({"icon_id": entry["id"], "kind": "near_twin",
                          "alteration": f"{method}: {desc}", "twin_change": list(change)})
            n_twin_done += 1
            placed = True
            break
        if not placed:
            continue
    # Preserve the requested difficulty: resample instead of dropping near-twins.
    if n_twin_done != n_twin:
        return None

    # --- fresh distinct distractors ---
    for _ in range(n_fresh):
        got = fresh_icon()
        if got is None:
            return None
        e, _, t = got
        ta = displayed_array(t)
        if not differs_from_all(ta):
            return None  # extremely unlikely with distinct, non-degenerate ids
        tiles.append(t)
        arrs.append(ta)
        groups.append(-1)
        metas.append({"icon_id": e["id"], "kind": "single", "alteration": None,
                      "twin_change": None})

    if len(tiles) != M:
        return None

    # --- shuffle positions; reject layouts where a pair sits adjacent ---
    cols = grid_cols(M)
    order = list(range(M))
    for _ in range(60):
        rng.shuffle(order)
        if not _has_adjacent_pair(order, groups, cols, M):
            break
    tiles = [tiles[i] for i in order]
    groups = [groups[i] for i in order]
    metas = [metas[i] for i in order]
    return tiles, groups, metas


def grid_cols(M):
    if M <= 9:
        return 3
    if M in (10, 11, 12):
        return 4
    return 5  # 13..25; retain readable icon sizes at <=588 px


def _has_adjacent_pair(order, groups, cols, M):
    """True if any two members of one pair-group are orthogonally adjacent."""
    pos = {}  # group_id -> list of grid (r,c)
    for slot, src in enumerate(order):
        g = groups[src]
        if g < 0:
            continue
        r, c = divmod(slot, cols)
        pos.setdefault(g, []).append((r, c))
    for cells in pos.values():
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                (r1, c1), (r2, c2) = cells[i], cells[j]
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return True
    return False


# --------------------------------------------------------------------------- layout/draw
# The format pattern uses letters (a-b,c-d): a numeric example such as "1-7,2-9"
# would often match a real gold pair.  The
# "pictures" opening says "The image shows" (the item noun stays "pictures").
# Every template carries the same explicit rules (where
# the numbers are, what "the same" excludes, that groups are pairs, and the
# exact answer format), so no template gives its own format hint or leaves open
# whether a rotated or mirrored copy counts as the same.  The template is still drawn after the
# scene, so the scene for a given RNG is unchanged.
QUESTION_TEMPLATES = [
    ("items", "There are {M} items, numbered 1 to {M}. Find the identical pairs ({rule}). {fmt}"),
    ("icons", "Among the {M} numbered icons, find every identical pair ({rule}). {fmt}"),
    ("pictures", "The image shows {M} numbered pictures. List every pair of identical pictures ({rule}). {fmt}"),
]
QUESTION_RULE = (
    'identical means the same shape, parts, colour and orientation, so rotated, mirrored or '
    'edited copies differ; identical items come only in pairs'
)
QUESTION_FORMAT = 'Answer each pair as a-b, separated by commas, in any order.'


def question_text(rng, M):
    noun, head = rng.choice(QUESTION_TEMPLATES)
    return head.format(M=M, rule=QUESTION_RULE, fmt=QUESTION_FORMAT)


def layout_geom(size, M):
    """Compute grid geometry. A label strip is reserved at the TOP of every cell
    so the index number never overlaps the icon -- this is what lets the saved
    PNG be cropped to a label-free icon region that is byte-identical for twins.
    Returns (cols, rows, cell, gap, ox, oy, label_h, tile, tile_off)."""
    cols = grid_cols(M)
    rows = math.ceil(M / cols)
    margin = int(size * 0.04)
    gap = int(size * 0.012)
    avail = size - 2 * margin
    cell = min((avail - (cols - 1) * gap) // cols,
               (avail - (rows - 1) * gap) // rows)
    label_h = max(15, cell // 5)                 # reserved top strip for the number
    inner = cell - label_h                       # space for the icon, below label
    tile = int(inner * 0.92)
    tile_off = label_h + (inner - tile) // 2     # icon vertically centered below strip
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    ox = (size - grid_w) // 2
    oy = (size - grid_h) // 2
    return cols, rows, cell, gap, ox, oy, label_h, tile, tile_off


def make_problem(lib, rng, size, diff):
    size = extra.resolve_size(size, diff)

    # style for the WHOLE problem; silhouette shares one tint
    style = "silhouette" if rng.random() < 0.6 else "color"
    color = rng.choice(SIL_COLORS) if style == "silhouette" else None

    res = None
    for _ in range(40):
        M, n_pairs, n_twin, strength = diff_params(diff, rng)
        _, _, _, _, _, _, _, tile, _ = layout_geom(size, M)
        res = build_items(lib, rng, M, n_pairs, n_twin, strength, tile, style, color)
        if res is not None:
            break
    if res is None:
        raise RuntimeError(f"could not build readable {diff} pairs at size={size}")
    tiles, groups, metas = res
    M = len(tiles)
    cols, rows, cell, gap, ox, oy, label_h, tile, tile_off = layout_geom(size, M)

    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    font = load_bundled_font(max(12, label_h - 4), 'sans-bold')
    slot_boxes = {}
    icon_off_x = (cell - tile) // 2              # icons horizontally centered

    for i in range(M):
        slot = i + 1
        r, c = divmod(i, cols)
        x0 = ox + c * (cell + gap)
        y0 = oy + r * (cell + gap)
        box = (x0, y0, x0 + cell, y0 + cell)
        slot_boxes[slot] = box
        d.rectangle(box, outline=(228, 222, 214), width=1)
        t = tiles[i]
        img.paste(t, (x0 + icon_off_x, y0 + tile_off), t)
        # index label in the reserved top strip (never overlaps the icon)
        d.text((x0 + 4, y0 + 2), str(slot), fill=(15, 15, 15), font=font)

    # --- COMPUTE ground truth from the placed tiles (byte-identical groups) ---
    pairs = pairs_from_tiles(tiles)

    # sanity: must equal intended groups
    intended = intended_pairs(groups)
    assert pairs == intended, f"pair mismatch: {pairs} vs intended {intended}"

    answer = ",".join(f"{a}-{b}" for a, b in pairs)
    qt = question_text(rng, M)

    n_near = sum(1 for m in metas if m["kind"] == "near_twin")

    weakest = None
    shown = [displayed_array(t) for t in tiles]
    for i in range(M):
        for j in range(i + 1, M):
            if (i + 1, j + 1) in pairs:
                continue
            metrics = array_metrics(shown[i], shown[j])
            assert metrics["readable"]
            assert pose_distinct(shown[i], shown[j]), \
                f"items {i + 1} and {j + 1} are the same picture up to a small shift"
            if weakest is None or metrics["component_pixels"] < weakest["component_pixels"]:
                weakest = {"indices": [i + 1, j + 1], **metrics}
    meta = {
        "generator_version": GENERATOR_VERSION,
        "readability": {"min_channel_contrast": MIN_CONTRAST,
                        "min_component_pixels": MIN_COMPONENT_PIXELS,
                        "min_component_core_pixels": MIN_COMPONENT_CORE,
                        "pose_search_px": POSE_SEARCH,
                        "twin_max_change": list(TWIN_MAX_CHANGE),
                        "weakest_nonmatching_pair": weakest},
        "difficulty": diff,
        "style": style,
        "color": list(color) if color else None,
        "M": M,
        "n_pairs": len(pairs),
        "n_near_twin": n_near,
        "rows": rows,
        "cols": cols,
        "pairs": pairs,
        "icon_ids": [m["icon_id"] for m in metas],          # per index (1..M)
        "item_kinds": [m["kind"] for m in metas],
        "alterations": [m["alteration"] for m in metas],
        # near-twins: [pixels changed vs its own pair, that pair's ink pixels]
        "twin_change": [m["twin_change"] for m in metas],
        "slot_boxes": {str(k): list(v) for k, v in slot_boxes.items()},
        # per-cell icon-only bbox (label-free), x0,y0 relative to the cell box:
        "tile_px": tile, "cell_px": cell,
        "icon_off": [icon_off_x, tile_off],
        "icon_bbox": {str(k): [v[0] + icon_off_x, v[1] + tile_off,
                               v[0] + icon_off_x + tile, v[1] + tile_off + tile]
                      for k, v in slot_boxes.items()},
    }
    return img, qt, answer, meta


def pairs_from_tiles(tiles):
    """Group items by identical displayed RGB pixels -> sorted list of (a,b) pairs.

    Asserts every duplicate group has size exactly 2 (no accidental triple).
    """
    by_bytes = {}
    for i, t in enumerate(tiles):
        by_bytes.setdefault(displayed_tile(t).tobytes(), []).append(i + 1)  # 1-indexed
    pairs = []
    for idxs in by_bytes.values():
        if len(idxs) == 1:
            continue
        assert len(idxs) == 2, f"group of size {len(idxs)}: {idxs}"
        a, b = sorted(idxs)
        pairs.append((a, b))
    pairs.sort()
    return pairs


def intended_pairs(groups):
    by_group = {}
    for i, g in enumerate(groups):
        if g >= 0:
            by_group.setdefault(g, []).append(i + 1)
    pairs = []
    for idxs in by_group.values():
        assert len(idxs) == 2
        pairs.append(tuple(sorted(idxs)))
    pairs.sort()
    return pairs

