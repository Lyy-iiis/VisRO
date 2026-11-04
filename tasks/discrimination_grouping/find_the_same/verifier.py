"""Independent verifier for find_the_same.

It never imports the generator, iconlib or the adapter.
Everything is recomputed from the delivered PNG, with the record's meta used
only as a hypothesis that the pixels must confirm:

  grid      every cell outline is decoded from the image as a closed 1-px ring
            of the outline colour; the rings, sorted row-major, must be exactly
            meta.slot_boxes, M of them, laid out cols-wide as the tier says;
  labels    the number in each box's top strip is decoded by template matching
            against DejaVuSans-Bold (assets/fonts, the only font in the repo)
            and must read 1..M row-major;
  layout    inside each box, every pixel outside the label strip and outside
            the icon box is the background colour, and every icon box has the
            same size and offset in its cell -- so comparing icon crops
            compares the whole visible item;
  gold      icon crops are grouped by exact RGB bytes; every group must have
            size 1 or 2, and the size-2 groups must be exactly the answer;
  margin    every other pair of crops must differ by a connected patch of
            >= 48 px whose max-channel difference is >= 64, with >= 12 px left
            after a 3x3 erosion (the same floor the question's tier promises),
            measured on the delivered pixels;
  pose      the same floor must also hold after aligning the two crops by
            their best translation (fewest >= 64 pixels) within 2 px of their
            ink-box offset, so no distractor is a pair member merely moved
            inside its box (a hidden third copy under the stated rules);
  roles     each near-twin must be nearest (changed px / pair ink px) to the
            pair whose icon it declares, and change at most 55% of that pair's
            ink pixels, as meta.twin_change states;
  form      the question must be one of the three declared templates with the
            right M and the full rules sentence; its format pattern must share
            no component with the gold; the answer must be the canonical
            "a-b,c-d" string; the item must match its tier row.
"""
from __future__ import annotations

import functools
import itertools
import os
import re

# Importing numpy with OpenBLAS's default pool spins one thread per core (240
# here, ~15 CPU-s per import) and nothing in this task uses BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import ndimage  # noqa: E402

from shared.drawing.fonts import load_bundled_font  # noqa: E402

CORRECTNESS = ("Decodes the grid (outline rings), the 1..M number labels (DejaVuSans-Bold template "
               "match) and every item crop from the delivered PNG, checks nothing else is drawn in a "
               "box, groups crops by exact RGB bytes and requires the identical groups to be exactly "
               "the answer pairs (meta.pairs and the 'pair' roles must agree; each near-twin must be "
               "pixel-nearest to its declared pair and change <=55% of its ink).  The question's "
               "format pattern (letters a-b,c-d) must share no pair, number or prefix with the gold.")
UNIQUENESS = ("Every identical group found in the pixels has exactly two members (no triple), and every "
              "pair outside the answer differs on the delivered PNG by a connected patch of >=48 px with "
              "max-channel difference >=64 and >=12 px after 3x3 erosion, both as placed and after the "
              "best translation within 2 px of their ink-box offset, so no second pair -- not even a "
              "shifted copy -- can be called identical; the question states that rotated/mirrored/"
              "recoloured/part-changed copies are different and that groups are pairs.")

BG = (252, 250, 246)
OUTLINE = (228, 222, 214)
SIL_TINTS = {(30, 30, 30), (25, 60, 120), (120, 35, 35), (30, 95, 55),
             (70, 45, 105), (140, 80, 20), (20, 90, 100)}
CONTRAST, AREA, CORE = 64, 48, 12
INK = 32              # max-channel distance from BG that counts as ink
POSE = 2              # px of slack around the ink-box offset (and ink-box size) when aligning
TWIN_CAP = (55, 100)  # a near-twin changes <= 55/100 of its pair's ink pixels

# tier -> (size, allowed M, (min, max) identical pairs, (min, max) near-twins)
TIER_SPEC = {
    "easy": (448, (9, 10, 11), (2, 3), (0, 0)),
    "medium": (448, (11, 12, 14), (3, 4), (1, 2)),
    "hard": (448, (14, 15, 16), (4, 5), (2, 3)),
    "xhard": (588, (20,), (6, 6), (6, 6)),
    "xxhard": (588, (25,), (8, 8), (8, 8)),
}
# Declared layout per (image size, M): (cell side, icon tile side, icon x offset,
# icon y offset) inside the cell box, i.e. the tile sizes the tier table states.
# Pinning it makes a shifted meta.icon_boxes fail even where the shifted box
# would still hold all of the icon's ink.
LAYOUT = {
    (448, 9): (134, 99, 17, 30),
    **{(448, m): (99, 73, 13, 22) for m in (10, 11, 12)},
    **{(448, m): (78, 57, 10, 18) for m in (14, 15, 16)},
    **{(588, m): (102, 75, 13, 23) for m in (20, 25)},
}

_HEADS = {
    "items": "There are {M} items, numbered 1 to {M}. Find the identical pairs ({rule}). {fmt}",
    "icons": "Among the {M} numbered icons, find every identical pair ({rule}). {fmt}",
    "pictures": "The image shows {M} numbered pictures. List every pair of identical pictures ({rule}). {fmt}",
}
_RULE = (
    'identical means the same shape, parts, colour and orientation, so rotated, mirrored or '
    'edited copies differ; identical items come only in pairs'
)
_FORMAT = 'Answer each pair as a-b, separated by commas, in any order.'
ANSWER_RE = re.compile(r"[1-9]\d*-[1-9]\d*(,[1-9]\d*-[1-9]\d*)*")
EXAMPLE_RE = re.compile(r"Answer each pair as (\S+), separated by commas")


def expected_questions(M: int) -> set[str]:
    return {h.format(M=M, rule=_RULE, fmt=_FORMAT) for h in _HEADS.values()}


def example_problems(question: str, gold) -> list[str]:
    """The format pattern printed in the question must not give away any gold.

    It must be letters only, and must share no whole string, pair (either
    order), number or leading number with the answer; no 'n-m' written
    anywhere in the question may be a gold pair either.
    """
    P = []
    m = EXAMPLE_RE.search(question or "")
    if not m:
        return ["question has no format pattern"]
    ex = m.group(1)
    gold_pairs = {frozenset(map(str, p)) for p in gold}
    gold_nums = {str(x) for p in gold for x in p}
    answer = ",".join(f"{a}-{b}" for a, b in gold)
    ex_pairs = [p for p in ex.split(",") if p]
    ex_nums = {x for p in ex_pairs for x in p.split("-")}
    if re.search(r"\d", ex):
        P.append(f"format pattern {ex!r} contains a number")
    if ex == answer or {frozenset(p.split("-")) for p in ex_pairs} & gold_pairs or ex_nums & gold_nums \
            or answer.split("-")[0] == ex.split("-")[0]:
        P.append(f"format pattern {ex!r} shares a component with the gold {answer!r}")
    if {frozenset(t) for t in re.findall(r"(\d+)\s*-\s*(\d+)", question)} & gold_pairs:
        P.append("the question text writes out a gold pair")
    return P


def ink_box(c: np.ndarray):
    ys, xs = np.nonzero(np.abs(c.astype(np.int16) - np.array(BG, np.int16)).max(axis=2) >= INK)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def moved(c: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Crop translated by (dy, dx); uncovered pixels are background."""
    out = np.empty_like(c)
    out[:] = np.array(BG, c.dtype)
    H, W = c.shape[:2]
    if abs(dy) < H and abs(dx) < W:
        out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
            c[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def aligned_difference(a: np.ndarray, b: np.ndarray, box_a, box_b):
    """Best translation of b onto a (fewest pixels differing by >= CONTRAST)
    among (0, 0) and the shifts within POSE px of the ink-box offset; None when
    the ink boxes differ in size by more than POSE px (then no shift aligns)."""
    if box_a is None or box_b is None:
        return None
    if abs((box_a[2] - box_a[0]) - (box_b[2] - box_b[0])) > POSE or \
            abs((box_a[3] - box_a[1]) - (box_b[3] - box_b[1])) > POSE:
        return None
    oy, ox = box_a[0] - box_b[0], box_a[1] - box_b[1]
    a16 = a.astype(np.int16)
    best = None
    for dy, dx in {(0, 0)} | {(oy + i, ox + j) for i in range(-POSE, POSE + 1) for j in range(-POSE, POSE + 1)}:
        n = int((np.abs(a16 - moved(b, dy, dx).astype(np.int16)).max(axis=2) >= CONTRAST).sum())
        key = (n, abs(dy) + abs(dx), dy, dx)
        if best is None or key < best:
            best = key
    n, _, dy, dx = best
    return {"shift": (dy, dx), "contrast_px": n, **pair_difference(a, moved(b, dy, dx))}


def change_px(a: np.ndarray, base: np.ndarray):
    """(pixels of a differing from base by >= CONTRAST, ink pixels of base)."""
    d = np.abs(a.astype(np.int16) - base.astype(np.int16)).max(axis=2) >= CONTRAST
    ink = np.abs(base.astype(np.int16) - np.array(BG, np.int16)).max(axis=2) >= INK
    return int(d.sum()), int(ink.sum())


def grid_cols(M: int) -> int:
    return 3 if M <= 9 else 4 if M <= 12 else 5


def parse_answer(text: str):
    if not isinstance(text, str) or not ANSWER_RE.fullmatch(text):
        return None
    return [tuple(int(x) for x in p.split("-")) for p in text.split(",")]


def pair_difference(a: np.ndarray, b: np.ndarray, contrast=CONTRAST, area=AREA, core=CORE) -> dict:
    """Largest qualifying connected difference patch between two RGB crops."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    mask = d >= contrast
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return {"readable": False, "area": 0, "core": 0, "max_diff": int(d.max())}
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    er = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))
    cores = np.bincount(lab[er], minlength=n + 1)
    ok = [k for k in range(1, n + 1) if areas[k] >= area and cores[k] >= core]
    k = max(ok, key=lambda i: areas[i]) if ok else int(np.argmax(areas[1:])) + 1
    return {"readable": bool(ok), "area": int(areas[k]), "core": int(cores[k]), "max_diff": int(d.max())}


def decode_rings(arr: np.ndarray, min_side: int = 30):
    """Closed 1-px square rings of the outline colour -> sorted [x0, y0, x1, y1] (inclusive)."""
    mask = np.all(arr == np.array(OUTLINE, np.uint8), axis=2)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    rings = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if h != w or h < min_side:
            continue
        sub = lab[sl] == k
        ring = np.zeros_like(sub)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        if np.array_equal(sub, ring):
            rings.append([xs.start, ys.start, xs.stop - 1, ys.stop - 1])
    return sorted(rings, key=lambda b: (b[1], b[0]))


@functools.lru_cache(maxsize=4096)
def _label_template(text: str, size: int, fill: tuple):
    im = Image.new("RGB", (size * 4 + 8, size * 2 + 8), BG)
    ImageDraw.Draw(im).text((4, 2), text, fill=fill, font=load_bundled_font(size, 'sans-bold'))
    a = np.asarray(im)
    ink = np.any(a != np.array(BG, np.uint8), axis=2)
    ys, xs = np.nonzero(ink)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _ink_crop(region: np.ndarray):
    ink = np.any(region != np.array(BG, np.uint8), axis=2)
    if not ink.any():
        return None, None
    ys, xs = np.nonzero(ink)
    return region[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (int(xs.min()), int(ys.min()))


def read_labels(arr: np.ndarray, boxes, icon_top_rows, M: int):
    """Decode each box's number by exact template match.  Returns (labels, size, problems)."""
    crops, problems = [], []
    for k, (x0, y0, x1, y1) in enumerate(boxes, 1):
        crop, _ = _ink_crop(arr[y0 + 1:icon_top_rows[k - 1], x0 + 1:x1])
        if crop is None:
            problems.append(f"slot {k}: no number label in its top strip")
        crops.append(crop)
    if problems:
        return None, None, problems
    first = crops[0]
    fill = tuple(int(v) for v in first.reshape(-1, 3)[first.reshape(-1, 3).sum(1).argmin()])
    if sum(fill) > 3 * 60:
        return None, None, [f"label ink too light {fill}"]
    size = next((s for s in range(8, 49) if np.array_equal(_label_template("1", s, fill), first)), None)
    if size is None:
        return None, None, ["slot 1: label does not read as '1' in DejaVuSans-Bold at any size 8-48"]
    labels = []
    for k, crop in enumerate(crops, 1):
        hits = [n for n in range(1, max(M, 30) + 1)
                if np.array_equal(_label_template(str(n), size, fill), crop)]
        labels.append(hits[0] if len(hits) == 1 else None)
        if hits != [k]:
            problems.append(f"slot {k}: label reads as {hits or 'nothing'}")
    return labels, size, problems


def analyse(record: dict, image) -> dict:
    """All checks; returns {"problems": [...], **measurements}."""
    P: list[str] = []
    out = {"problems": P}
    meta = record.get("meta") or {}
    tier = record.get("tier")
    if tier not in TIER_SPEC:
        P.append(f"unknown tier {tier!r}")
        return out
    size, Ms, (pmin, pmax), (tmin, tmax) = TIER_SPEC[tier]

    # ---- record / question / answer form
    if record.get("answer_type") != "pair-list":
        P.append(f"answer_type {record.get('answer_type')!r} != 'pair-list'")
    if record.get("choices") is not None:
        P.append("free-form task must not carry choices")
    gold = parse_answer(record.get("answer"))
    if gold is None:
        P.append(f"answer {record.get('answer')!r} is not 'a-b,c-d'")
        return out
    if any(a >= b for a, b in gold) or gold != sorted(gold):
        P.append("answer is not canonical (a<b within pairs, pairs sorted)")
    members = [x for p in gold for x in p]
    if len(members) != len(set(members)):
        P.append("answer pairs share a member")
    try:
        M = int(meta["M"])
        boxes = [list(map(int, b)) for b in meta["slot_boxes"]]
        iboxes = [list(map(int, b)) for b in meta["icon_boxes"]]
        kinds, ids = list(meta["item_kinds"]), list(meta["icon_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        P.append(f"meta incomplete: {exc!r}")
        return out
    if record.get("question") not in expected_questions(M):
        P.append("question text is not one of the declared templates for this M")
    P += example_problems(record.get("question"), gold)
    if M not in Ms:
        P.append(f"M={M} outside tier {tier} {Ms}")
    if not all(1 <= x <= M for x in members):
        P.append("answer index outside 1..M")
    if [list(p) for p in gold] != [list(p) for p in meta.get("pairs", [])]:
        P.append("meta.pairs != answer")
    if not (len(boxes) == len(iboxes) == len(kinds) == len(ids) == M):
        P.append("meta per-slot lists do not have M entries")
        return out

    # ---- image and grid
    arr = np.asarray(image.convert("RGB"))
    H, W = arr.shape[:2]
    if (W, H) != (size, size) or meta.get("size") != size:
        P.append(f"image {W}x{H} (meta size {meta.get('size')}) != tier size {size}")
    rings = decode_rings(arr)
    out["rings"] = len(rings)
    if rings != boxes:
        P.append(f"decoded {len(rings)} outline rings != meta.slot_boxes ({M})")
        return out
    cols = grid_cols(M)
    rows_y = sorted({b[1] for b in boxes})
    cols_x = sorted({b[0] for b in boxes})
    if len(cols_x) != cols or len(rows_y) != -(-M // cols) or \
            any(boxes[k - 1][:2] != [cols_x[(k - 1) % cols], rows_y[(k - 1) // cols]] for k in range(1, M + 1)):
        P.append("boxes are not a row-major grid of the tier's column count")
    if meta.get("cols") != cols or meta.get("rows") != len(rows_y):
        P.append("meta rows/cols disagree with the decoded grid")
    outside = np.ones(arr.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        outside[y0:y1 + 1, x0:x1 + 1] = False
    if np.any(arr[outside] != np.array(BG, np.uint8)):
        P.append("ink outside the numbered boxes")

    # ---- icon boxes: same offset and size in every cell, strictly inside the box
    offs = {(ib[0] - b[0], ib[1] - b[1], ib[2] - ib[0], ib[3] - ib[1]) for b, ib in zip(boxes, iboxes)}
    if len(offs) != 1:
        P.append("icon boxes differ in size or offset between cells")
        return out
    ox, oy, tw, th = offs.pop()
    cell = boxes[0][2] - boxes[0][0]
    if not (tw == th == meta.get("tile_px") and 0 < ox and ox + tw < cell and 1 < oy and oy + th < cell):
        P.append("icon box not strictly inside its cell / tile size mismatch")
        return out
    if LAYOUT.get((size, M)) != (cell, tw, ox, oy):
        P.append(f"layout (cell {cell}, tile {tw}, offset {ox},{oy}) is not the declared one for M={M}")

    # ---- labels
    labels, fsize, lp = read_labels(arr, boxes, [b[1] + oy for b in boxes], M)
    P += lp
    out["label_px"] = fsize

    # ---- nothing but background outside the icon box below the label strip
    bg = np.array(BG, np.uint8)
    crops = []
    for k, ((x0, y0, x1, y1), (ix0, iy0, ix1, iy1)) in enumerate(zip(boxes, iboxes), 1):
        region = arr[iy0:y1, x0 + 1:x1].copy()
        region[0:iy1 - iy0, ix0 - x0 - 1:ix1 - x0 - 1] = bg
        if np.any(region != bg):
            P.append(f"slot {k}: ink outside its icon box")
        crop = arr[iy0:iy1, ix0:ix1]
        ink = (np.abs(crop.astype(np.int16) - bg).max(axis=2) >= 32).sum() / float(tw * th)
        if not 0.03 <= ink <= 0.92:
            P.append(f"slot {k}: icon footprint {ink:.3f} outside [0.03, 0.92]")
        crops.append(crop)

    # ---- gold from pixels
    groups: dict[bytes, list[int]] = {}
    for k, c in enumerate(crops, 1):
        groups.setdefault(c.tobytes(), []).append(k)
    big = [g for g in groups.values() if len(g) > 2]
    if big:
        P.append(f"identical groups larger than a pair: {big}")
    pix_pairs = sorted(tuple(g) for g in groups.values() if len(g) == 2)
    out["pixel_pairs"] = pix_pairs
    if pix_pairs != gold:
        P.append(f"identical crops in the image {pix_pairs} != answer {gold}")
    if not pmin <= len(pix_pairs) <= pmax:
        P.append(f"{len(pix_pairs)} identical pairs outside tier range {pmin}-{pmax}")

    # ---- uniqueness margin: every other pair is visibly different
    gold_set = set(gold)
    weakest = None
    boxes_ink = [ink_box(c) for c in crops]
    pose_px = []
    for a, b in itertools.combinations(range(1, M + 1), 2):
        if (a, b) in gold_set:
            continue
        m = pair_difference(crops[a - 1], crops[b - 1])
        if not m["readable"]:
            P.append(f"items {a} and {b} are not in the answer but differ only by "
                     f"{m['area']} px / core {m['core']} (max diff {m['max_diff']})")
        if weakest is None or m["area"] < weakest[0]:
            weakest = (m["area"], m["core"], a, b)
        al = aligned_difference(crops[a - 1], crops[b - 1], boxes_ink[a - 1], boxes_ink[b - 1])
        if al is not None and al["shift"] != (0, 0):
            pose_px.append(al["contrast_px"])
            if not al["readable"]:
                P.append(f"items {a} and {b} differ only by a shift: aligned by {al['shift']} their largest "
                         f"difference patch is {al['area']} px / core {al['core']} "
                         f"({al['contrast_px']} px >= {CONTRAST}), below the floor")
    out["weakest"] = weakest
    out["pose_aligned_min_px"] = min(pose_px) if pose_px else None
    w = (meta.get("readability") or {}).get("weakest_nonmatching_pair") or {}
    if weakest and w.get("component_pixels") is not None and w.get("component_pixels") != weakest[0]:
        P.append(f"meta weakest pair area {w.get('component_pixels')} != measured {weakest[0]}")

    # ---- roles in meta agree with the pixels and the tier
    pair_slots = {x for p in pix_pairs for x in p}
    if {k for k in range(1, M + 1) if kinds[k - 1] == "pair"} != pair_slots:
        P.append("meta item_kinds 'pair' slots != identical crops")
    for a, b in pix_pairs:
        if ids[a - 1] != ids[b - 1]:
            P.append(f"pair {a}-{b} has different icon ids")
    pair_ids = [ids[a - 1] for a, _ in pix_pairs]
    if len(set(pair_ids)) != len(pair_ids):
        P.append("two identical pairs use the same icon")
    twins = [k for k in range(1, M + 1) if kinds[k - 1] == "near_twin"]
    singles = [k for k in range(1, M + 1) if kinds[k - 1] == "single"]
    if len(twins) + len(singles) + len(pair_slots) != M:
        P.append("unknown item kinds")
    if not (tmin <= len(twins) <= tmax) or meta.get("n_near_twin") != len(twins):
        P.append(f"{len(twins)} near-twins outside tier range {tmin}-{tmax} or != meta")
    if any(ids[k - 1] not in pair_ids for k in twins):
        P.append("a near-twin is not derived from a pair icon")
    single_ids = [ids[k - 1] for k in singles]
    if len(set(single_ids)) != len(single_ids) or set(single_ids) & set(pair_ids):
        P.append("single distractors repeat an icon")
    # a near-twin must be the pixel-nearest to the pair it claims to alter and
    # change at most TWIN_CAP of that pair's ink (compared as integers)
    tc = meta.get("twin_change")
    if not isinstance(tc, list) or len(tc) != M:
        P.append("meta.twin_change missing or not per slot")
        tc = [None] * M
    twin_fracs = []
    for k in range(1, M + 1):
        if k not in twins:
            if tc[k - 1] is not None:
                P.append(f"slot {k}: twin_change set on a {kinds[k - 1]} item")
            continue
        base = next((a for a, _ in pix_pairs if ids[a - 1] == ids[k - 1]), None)
        if base is None:
            continue
        own = change_px(crops[k - 1], crops[base - 1])
        twin_fracs.append(own[0] / max(1, own[1]))
        if list(own) != tc[k - 1]:
            P.append(f"slot {k}: meta.twin_change {tc[k - 1]} != measured {list(own)}")
        if own[0] * TWIN_CAP[1] > TWIN_CAP[0] * own[1]:
            P.append(f"near-twin {k} changes {own[0]}/{own[1]} of pair {base}'s ink (> 55%)")
        for a, _ in pix_pairs:
            if a == base:
                continue
            other = change_px(crops[k - 1], crops[a - 1])
            if own[0] * other[1] > other[0] * own[1]:
                P.append(f"near-twin {k} is nearer to pair {a} than to its declared pair {base}")
                break
    out["twin_fracs"] = twin_fracs

    # ---- no pair members orthogonally adjacent (tier rule)
    pos = {k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, M + 1)}
    for a, b in pix_pairs:
        if abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1:
            P.append(f"pair {a}-{b} is orthogonally adjacent")

    # ---- style: a silhouette grid is one tint over the background.  Resampled
    # edges scale the tint's brightness (LANCZOS ringing, transparent pixels
    # stored as black), so an edge pixel is a*tint + b*background: every pixel
    # of a pair or single item must lie on the plane spanned by the tint and
    # the background colour.  Near-twins are left out: a recoloured copy is a
    # legitimate difference the question names.
    style, color = meta.get("style"), meta.get("color")
    if style == "silhouette":
        if not color or tuple(color) not in SIL_TINTS:
            P.append(f"silhouette grid with tint {color!r}")
        else:
            # px = u*tint + w*background with u, w >= 0 (a cone, not just the
            # plane: the plane through a tint like (140,80,20) also holds its
            # channel-reversed colour).  Elementwise 2x2 solve, no BLAS threads.
            C, B = np.array(color, float), np.array(BG, float)
            px = np.concatenate([crops[k - 1].reshape(-1, 3) for k in range(1, M + 1)
                                 if kinds[k - 1] != "near_twin"]).astype(float)
            n = np.cross(C, B)
            n /= np.sqrt((n * n).sum())
            resid = float(np.abs(px[:, 0] * n[0] + px[:, 1] * n[1] + px[:, 2] * n[2]).max())
            cc, cb, bb = (C * C).sum(), (C * B).sum(), (B * B).sum()
            pc = px[:, 0] * C[0] + px[:, 1] * C[1] + px[:, 2] * C[2]
            pb = px[:, 0] * B[0] + px[:, 1] * B[1] + px[:, 2] * B[2]
            det = cc * bb - cb * cb
            u, w = (bb * pc - cb * pb) / det, (cc * pb - cb * pc) / det
            # a gray tint is nearly parallel to the background: the split into
            # u and w is ill-conditioned there, so only the plane test applies
            gray = len(set(color)) == 1
            out["tint_residual"] = resid
            if resid > 2.5 or (not gray and (u.min() < -0.03 or w.min() < -0.03)):
                P.append(f"silhouette items are not one tint over the background (residual {resid:.1f})")
    elif style == "color":
        if color is not None:
            P.append("colour grid carries a tint")
    else:
        P.append(f"unknown style {style!r}")
    return out


def verify(record: dict, image) -> list[str]:
    return analyse(record, image)["problems"]
