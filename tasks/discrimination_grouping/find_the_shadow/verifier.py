"""Independent verifier for find_the_shadow.

Imports nothing from tasks/discrimination_grouping/find_the_shadow/generator
or from the adapter.  What it knows that is not in the record:

  * the five tier rows (canvas, grid, L, clones per letter, answerless-family
    size) and the question template -- the DECLARED FORM the record must match;
  * the page layout every tier shares (MARGIN 5, GAP 4, LABEL_H 18, TILE 92,
    DejaVuSans-Bold 15 labels at (+6, +1), 2 px grey cell frames), which it
    CHECKS against the pixels (frames, white page) before using it;
  * the repository's icon library (iconlib + its SVG files), used only to
    re-rasterise the base icons the record names, so that the named paths --
    and the core_key built from them -- are tied to the pixels;
  * the release's silhouette-class table (SILHOUETTE_CLASSES: the 41 drawable
    icon files that draw the same silhouette as another one, in 18 classes),
    pinned by a sha256 that the adapter asserts for the table it derives from
    its gate tables.

What it recomputes:
  1. the gold string from the record's meta (which base icon each lettered
     cell shows, which numbered cell holds that icon's unedited silhouette);
  2. the gold string again from the PNG alone: every label is READ by glyph
     template matching, every cell is classed pattern/shadow by colour, every
     silhouette is re-extracted by per-channel alpha against the white page and
     compared with every pattern IN PLACE (no re-centring, no rescaling --
     the question says no shadow is rotated, mirrored or rescaled).
     The masks are whole silhouettes, white cut-outs (holes) included, which is
     what the question means by "same shape" ("Shape includes white cut-outs");
  3. uniqueness under the question's own rule.  For pattern i and shadow j,
     d(i,j) = radius of the largest disc inside XOR(pattern, shadow).  A shadow
     "departs from the pattern nowhere" iff d <= 1.5 px AND every pixel of the
     XOR lies within 1.5 px of the pattern's outline (antialiasing only: no
     detached speck, no 1-2 px bump or notch); it is legibly refuted iff
     d >= 6 px, or its in-place IoU is < 0.62 (a different icon).  Every pair
     must fall on one side of that gap, every pattern must have exactly one
     shadow on the valid side, and those shadows must be distinct.  Then the
     pairing is forced, the unmatched list is its complement, and the answer
     string is unique.  Two extra guards: no two shadows within 6 px of each
     other (IoU >= 0.62), and no OTHER shadow equal to a pattern rotated by
     90/180/270 degrees or mirrored (d <= 3 px), so the orientation convention
     the question states is never what decides an item;
  4. the record's bookkeeping: core_key is recomputed from the silhouette
     classes in meta.base_icons (icon files that draw the same silhouette share
     a class), and each class must be the one SILHOUETTE_CLASSES declares for
     its icon (the icon itself when it has no twin), so one silhouette set has
     exactly one key; every named icon must be a train-split icon of style
     "silhouette" in the library; the roles must agree with meta.cells; every
     named icon is re-rasterised from its SVG and must equal, in place, both
     the pattern tile that shows it and its unedited shadow tile (IoU >= 0.9,
     disc <= 2 px), and a class icon other than the icon itself must render as
     its twin (IoU >= 0.9, disc <= 1.5 px, the adapter's own twin rule);
  5. the question's format example shares no letter, number or pair with the
     gold (it is built from letters and numbers no tier can produce).
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib
import json
import re
from functools import lru_cache

import numpy as np


CORRECTNESS = ("The gold string is rebuilt twice: from meta (which lettered cell shows which base icon, "
               "which numbered cell holds that icon's unedited silhouette) and from the PNG alone (labels "
               "read by glyph matching, cells classed by colour, whole silhouettes -- white cut-outs included "
               "-- re-extracted by alpha against white and compared with every pattern in place); both must "
               "equal the record's answer.  core_key is recomputed from the silhouette classes in meta.base_icons, "
               "each of which must be the class the release's twin table declares for its icon; every named base "
               "icon is re-rasterised from its SVG and must equal the pattern tile and the unedited shadow tile "
               "that show it, and its class icon must render as its twin.")
UNIQUENESS = ("For every pattern exactly one shadow departs from it by at most 1.5 px (largest disc in the "
              "XOR of the whole silhouettes, cut-outs included) with every departing pixel within 1.5 px of the "
              "pattern's outline, and every other shadow by at least 6 px or is a different icon (IoU < 0.62), "
              "so the pairing is forced and the unmatched list is its complement; no shadow pair is within "
              "6 px, no other shadow matches a rotated or mirrored pattern, and the question's format example "
              "shares no letter, number or pair with the gold.")

# ---------------------------------------------------------------- declared form
# tier: (W, H, rows, cols, L, clones per letter, clones in the answerless family)
TIERS = {
    "easy": (644, 364, 3, 6, 5, 1, 1),
    "medium": (728, 364, 3, 7, 6, 1, 1),
    "hard": (728, 476, 4, 7, 6, 2, 2),
    "xhard": (728, 616, 5, 7, 6, 3, 3),
    "xxhard": (728, 616, 5, 7, 5, 4, 3),
}
# the text is compressed to at most 238 tokens (vicuna + BOS) / 213
# (PaliGemma), was 469 / 430, with every rule of 1.2.x still stated once.  The
# example keeps letters and numbers no tier can produce (Y, Z > F; 91-94 > 30).
EXAMPLE = "Y-93,Z-91;unmatched:92,94"
QUESTION = (
    'Each colored pattern A-{last} has exactly one same-shape dark shadow among 1-{N}, none '
    'shared, so {U} match nothing. Shadows are never rotated, mirrored or rescaled; holes count. '
    'Look-alikes differ by one small edge edit; the match differs nowhere. Answer pairs in letter'
    " order, then ';unmatched:' and the rest ascending, e.g. Y-93,Z-91;unmatched:92,94."
)
EXAMPLE_LEAD = "e.g. "
MARGIN, GAP, LABEL_H, TILE = 5, 4, 18, 92
FRAME = (212, 212, 212)

TRUE_MAX = 1.5      # px: "departs nowhere" (antialiasing only; real true pairs read 0 or 1.0)
# px: every pixel of a true pair's XOR lies this close to the pattern's outline.
# The 5x5 L2 distance transform takes no value between 1.4 and 2.0, so this
# means <= 1.4 px: real true pairs read 0, 1.0 or 1.4.  A band of 2.0 would let a
# radius-2 bump, 2x2 specks one pixel off the outline and most radius-1 bumps /
# notches through.
OUTLINE_BAND = 1.5
WRONG_MIN = 6.0     # px: a notch / bump a reader can put a finger on
IOU_FAR = 0.62      # in-place IoU below this: a different icon altogether
TWIN_MAX = 3.0      # px: a rotated / mirrored pattern this close to a shadow is a twin
BASE_IOU_MIN = 0.9  # a named base icon, re-rasterised, against the tile that shows it
BASE_DISC_MAX = 2.0
TWIN_IOU_MIN = 0.9   # a base icon against its silhouette class's icon: the adapter's
TWIN_DISC_MAX = 1.5  # TWIN_IOU / TWIN_DISC (real, this renderer: >= 0.912, <= 1.4 px)
ROLE_U, ROLE_LONE = "answerless family", "lone shadow"

# Silhouette classes: drawable icon files whose gate tables put them at IoU >=
# 0.9 with an inscribed XOR disc <= 1.5 px (27 pairs, 41 icons, 18 classes),
# as {class representative (smallest path): the other members}.  Every other
# icon is its own class.  The adapter derives the same table from its cache and
# refuses to build when its sha256 differs from SILHOUETTE_CLASS_SHA256.
SILHOUETTE_CLASSES = {
    "gameicons/aussiesim/card-2-clubs.svg": ("gameicons/aussiesim/card-2-diamonds.svg",
                                             "gameicons/aussiesim/card-2-spades.svg"),
    "gameicons/aussiesim/card-3-clubs.svg": ("gameicons/aussiesim/card-3-diamonds.svg",
                                             "gameicons/aussiesim/card-3-spades.svg"),
    "gameicons/aussiesim/card-5-clubs.svg": ("gameicons/aussiesim/card-5-diamonds.svg",),
    "gameicons/aussiesim/card-7-clubs.svg": ("gameicons/aussiesim/card-7-diamonds.svg",),
    "gameicons/aussiesim/card-jack-clubs.svg": ("gameicons/aussiesim/card-jack-diamonds.svg",
                                                "gameicons/aussiesim/card-jack-spades.svg"),
    "gameicons/aussiesim/card-king-clubs.svg": ("gameicons/aussiesim/card-king-diamonds.svg",
                                                "gameicons/aussiesim/card-king-spades.svg"),
    "gameicons/sbed/big-egg.svg": ("mdi/svg/egg.svg",),
    "gameicons/sbed/health-normal.svg": ("mdi/svg/gamepad-round.svg", "mdi/svg/hospital.svg"),
    "mdi/svg/alpha-h.svg": ("mdi/svg/signal-hspa.svg",),
    "mdi/svg/alpha-m.svg": ("mdi/svg/size-m.svg",),
    "mdi/svg/alpha-o.svg": ("mdi/svg/numeric-0.svg",),
    "mdi/svg/alpha-s.svg": ("mdi/svg/size-s.svg",),
    "mdi/svg/alpha-x.svg": ("mdi/svg/roman-numeral-10.svg",),
    "mdi/svg/ambulance.svg": ("mdi/svg/truck-plus.svg",),
    "mdi/svg/border-all-variant.svg": ("mdi/svg/square-outline.svg",),
    "mdi/svg/rhombus-medium.svg": ("mdi/svg/rhombus.svg",),
    "mdi/svg/triangle-down.svg": ("mdi/svg/triangle-small-down.svg",),
    "mdi/svg/triangle-small-up.svg": ("mdi/svg/triangle.svg",),
}
SILHOUETTE_CLASS = {m: r for r, ms in SILHOUETTE_CLASSES.items() for m in (r, *ms)}

# natural-rival rungs: base icons may be drawn in any of the 8 orientations of
# the square (orientation code, the generator's ORIENT_OPS: 0 upright, 1 flip
# left-right, 2 flip top-bottom, 3 rotate 180, 4 rotate 90 (counter-clockwise), 5
# rotate 270, 6 transpose, 7 transverse; "mirrored" = a reflection, codes 1/2/6/7),
# text glyphs upright only, and the key is the sorted set of IDENTITY classes
# (orientation-free: icons whose silhouettes are twins in some orientation are one
# class).  The class table and the glyph list are the release's
# find_the_shadow_verify_table.json next to this file, pinned by VERIFY_TABLE_SHA256.
LAW_TIERS = ("hard", "xhard", "xxhard")
VERIFY_TABLE = (_ROOT / 'tasks/discrimination_grouping/find_the_shadow/find_the_shadow_verify_table.json')
VERIFY_TABLE_SHA256 = "2d55a79d70821ba8be2d27bca30294d9b2de84b9be6a8d9af99afcdedf50d02f"
REFLECTIONS = (1, 2, 6, 7)


@lru_cache(maxsize=None)
def verify_table():
    p = VERIFY_TABLE
    raw = p.read_bytes()
    if VERIFY_TABLE_SHA256 and hashlib.sha256(raw).hexdigest() != VERIFY_TABLE_SHA256:
        raise RuntimeError(f"{p} is not the release's verify table")
    d = json.loads(raw)
    ident = {m: rep for rep, ms in d["classes"].items() for m in (rep, *ms)}
    return dict(ident=ident, glyph=set(d["glyph_upright_only"]))


def core_key_of_law(base_icons):
    v = dict(task="find_the_shadow", ident=sorted(b["silhouette_class"] for b in base_icons))
    s = json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()
SILHOUETTE_CLASS_SHA256 = "79814584d43bd918c7e280267afd74ccd6edd8bc9c2c466aa9734c7f4acdd2fe"


def class_table_sha256(table):
    """sha256 of a {path: class representative} table, canonical JSON."""
    s = json.dumps(dict(sorted(table.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


assert class_table_sha256(SILHOUETTE_CLASS) == SILHOUETTE_CLASS_SHA256, "SILHOUETTE_CLASSES was edited"


def tier_numbers(tier):
    W, H, rows, cols, L, C, UC = TIERS[tier]
    N = L + L * C + 1 + UC + 1          # true shadows + clones + answerless family + lone
    assert rows * cols == L + N, tier
    return dict(W=W, H=H, rows=rows, cols=cols, L=L, C=C, UC=UC, N=N, U=N - L)


def expected_question(tier):
    t = tier_numbers(tier)
    return QUESTION.format(rows=t["rows"], cols=t["cols"], cells=t["rows"] * t["cols"], L=t["L"],
                           last=chr(ord("A") + t["L"] - 1), N=t["N"], U=t["U"])


ANSWER_RE = re.compile(r"^([A-Z]-\d+(?:,[A-Z]-\d+)*);unmatched:(\d+(?:,\d+)*)$")


def parse_answer(ans, L, N, U):
    """-> ({letter: number}, [unmatched]) or raises ValueError."""
    m = ANSWER_RE.match(ans)
    if not m:
        raise ValueError(f"answer not in the stated format: {ans!r}")
    pairs = [p.split("-") for p in m.group(1).split(",")]
    letters = [p[0] for p in pairs]
    if letters != [chr(ord("A") + i) for i in range(L)]:
        raise ValueError(f"letters {letters} are not A..{chr(ord('A') + L - 1)} in order")
    tokens = [p[1] for p in pairs] + m.group(2).split(",")
    if any(tok != str(int(tok)) for tok in tokens):
        raise ValueError("leading zeros")
    nums = [int(p[1]) for p in pairs]
    um = [int(x) for x in m.group(2).split(",")]
    if um != sorted(um) or len(um) != U:
        raise ValueError(f"unmatched list not {U} numbers in increasing order")
    if sorted(nums + um) != list(range(1, N + 1)):
        raise ValueError(f"pairs + unmatched are not a partition of 1..{N}")
    return dict(zip(letters, nums)), um


def answer_components(ans):
    """The pieces an answer string is made of: its letters, its numbers, its
    letter-number pairs, its first pair and its first unmatched number."""
    m = ANSWER_RE.match(ans)
    if not m:
        raise ValueError(f"not an answer-shaped string: {ans!r}")
    pairs = m.group(1).split(",")
    um = m.group(2).split(",")
    letters = {p.split("-")[0] for p in pairs}
    numbers = {int(p.split("-")[1]) for p in pairs} | {int(x) for x in um}
    return dict(letters=letters, numbers=numbers, pairs=set(pairs), first_pair=pairs[0],
                first_unmatched=um[0])


def example_overlap(example, gold):
    """Everything the question's format example shares with the gold: [] when
    the example can neither equal nor partly equal it."""
    e, g = answer_components(example), answer_components(gold)
    out = []
    if example == gold:
        out.append("equal")
    for k in ("letters", "numbers", "pairs"):
        if e[k] & g[k]:
            out.append(f"{k} {sorted(e[k] & g[k])}")
    if e["first_pair"] == g["first_pair"]:
        out.append("first pair")
    if e["first_unmatched"] == g["first_unmatched"]:
        out.append("first unmatched number")
    return out


def question_example(q):
    """The format example printed in the question, or None."""
    if EXAMPLE_LEAD not in q or not q.endswith("."):
        return None
    return q.split(EXAMPLE_LEAD, 1)[1][:-1]


def format_answer(pairs, N):
    um = sorted(set(range(1, N + 1)) - set(pairs.values()))
    return (",".join(f"{k}-{pairs[k]}" for k in sorted(pairs)) + ";unmatched:"
            + ",".join(str(x) for x in um))


# ---------------------------------------------------------------- page geometry
def cell_boxes(W, H, rows, cols):
    """Reading-order list of (frame box, tile box)."""
    cw = (W - 2 * MARGIN - (cols - 1) * GAP) // cols
    ch = (H - 2 * MARGIN - (rows - 1) * GAP) // rows
    ox = (W - (cols * cw + (cols - 1) * GAP)) // 2
    oy = (H - (rows * ch + (rows - 1) * GAP)) // 2
    tx = (cw - TILE) // 2
    out = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = ox + c * (cw + GAP), oy + r * (ch + GAP)
            out.append(((x0, y0, x0 + cw, y0 + ch),
                        (x0 + tx, y0 + LABEL_H, x0 + tx + TILE, y0 + LABEL_H + TILE)))
    return out


def check_frames(rgb, boxes):
    """Every cell frame is drawn where the layout says, and the page between is white."""
    probs = []
    H, W = rgb.shape[:2]
    covered = np.zeros((H, W), bool)
    for k, ((x0, y0, x1, y1), _) in enumerate(boxes):
        edge = np.concatenate([rgb[y0, x0:x1 + 1], rgb[y1, x0:x1 + 1],
                               rgb[y0:y1 + 1, x0], rgb[y0:y1 + 1, x1]])
        ok = (np.abs(edge.astype(int) - FRAME).max(1) <= 2).mean()
        if ok < 0.98:
            probs.append(f"cell {k + 1}: frame not where the layout puts it ({ok:.2f})")
        covered[y0:y1 + 1, x0:x1 + 1] = True
    outside = rgb[~covered]
    if len(outside) and (outside.min(1) < 250).mean() > 0.001:
        probs.append("ink outside the cell grid")
    return probs


# ---------------------------------------------------------------- labels
@lru_cache(maxsize=None)
def _font():
    from shared.drawing.fonts import load_bundled_font
    return load_bundled_font(15, "sans-bold")


@lru_cache(maxsize=None)
def _glyph(s):
    from PIL import Image, ImageDraw
    im = Image.new("L", (48, 18), 255)
    ImageDraw.Draw(im).text((6, 1), s, fill=15, font=_font())
    return np.asarray(im, np.float32)[2:, 2:]


def read_label(gray, x0, y0, cands):
    """Best candidate, its mean abs error, and the runner-up's (grey levels)."""
    reg = gray[y0 + 2:y0 + 18, x0 + 2:x0 + 48].astype(np.float32)
    sc = sorted((float(np.abs(reg - _glyph(s)).mean()), s) for s in cands)
    return sc[0][1], sc[0][0], sc[1][0]


# ---------------------------------------------------------------- silhouettes
def alpha_mask(tile):
    """Coverage >= 50% from per-channel alpha against the white page, using the
    channel in which the fill colour is darkest.  Also returns the fill colour."""
    t = tile.astype(np.float32)
    ink = 255.0 - t
    core = ink.max(2) > 60
    if core.sum() < 30:
        return np.zeros(tile.shape[:2], bool), None
    ref = np.percentile(ink[core], 95, axis=0)
    ch = int(np.argmax(ref))
    a = ink[..., ch] / max(float(ref[ch]), 1.0)
    fill = 255.0 - ref
    return a > 0.5, fill


def disc(a, b):
    import cv2
    x = np.logical_xor(a, b).astype(np.uint8)
    if not x.any():
        return 0.0
    return float(cv2.distanceTransform(x, cv2.DIST_L2, 5).max())


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(max(u, 1))


def off_outline(p, s):
    """How far (px) the farthest pixel of XOR(p, s) lies from p's outline: 0 for
    no XOR, 1-1.4 for antialiasing slivers along the edge; a detached speck or a
    1-2 px bump / notch reads 2 or more."""
    import cv2
    x = np.logical_xor(p, s)
    if not x.any():
        return 0.0
    pp = np.pad(p, 1, constant_values=False).astype(np.uint8)
    din = cv2.distanceTransform(pp, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dout = cv2.distanceTransform(1 - pp, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return float(np.where(p, din, dout)[x].max())


def dihedral(m):
    """The seven non-identity rotations / mirrors of a square mask."""
    return [np.rot90(m, 1), np.rot90(m, 2), np.rot90(m, 3), m[:, ::-1], m[::-1, :], m.T,
            np.rot90(m, 2).T]


# ---------------------------------------------------------------- decode
def decode(image, tier):
    """PNG -> labels, classes and masks per cell.  Returns (cells, problems)."""
    t = tier_numbers(tier)
    rgb = np.asarray(image.convert("RGB"))
    gray = np.asarray(image.convert("L"))
    probs = []
    if (rgb.shape[1], rgb.shape[0]) != (t["W"], t["H"]):
        return None, [f"image is {rgb.shape[1]}x{rgb.shape[0]}, tier {tier} is {t['W']}x{t['H']}"]
    boxes = cell_boxes(t["W"], t["H"], t["rows"], t["cols"])
    probs += check_frames(rgb, boxes)
    cands = [chr(ord("A") + i) for i in range(26)] + [str(i) for i in range(1, 41)]
    cells = []
    for k, ((x0, y0, _, _), (a0, b0, a1, b1)) in enumerate(boxes):
        lab, err, err2 = read_label(gray, x0, y0, cands)
        if err > 0.5 or err2 < err + 1.0:
            probs.append(f"cell {k + 1}: label unreadable (best {lab!r} err {err:.1f}, runner-up {err2:.1f})")
        tile = rgb[b0:b1, a0:a1]
        m, fill = alpha_mask(tile)
        if fill is None or m.sum() < 150:
            probs.append(f"cell {k + 1}: tile is empty")
            cells.append(dict(label=lab, kind=None, mask=m))
            continue
        sat = float(fill.max() - fill.min())
        val = float(fill.max())
        if sat > 60 and val > 120:
            kind = "pattern"
        elif sat < 45 and val < 110:
            kind = "shadow"
        else:
            kind = None
            probs.append(f"cell {k + 1}: fill {fill.round().tolist()} is neither a vivid pattern nor a dark shadow")
        # the tile must be ONE flat colour composited over white (antialiased
        # edges included): anything else is overpainting or a second colour
        t32 = tile.astype(np.float32)
        ink = 255.0 - t32
        ref = 255.0 - fill
        ch = int(np.argmax(ref))
        a = np.clip(ink[..., ch] / max(float(ref[ch]), 1.0), 0.0, 1.0)
        pred = 255.0 - a[..., None] * ref
        err = np.abs(t32 - pred).max(2)
        touched = a > 0.05
        if touched.any() and (err[touched] > 24).mean() > 0.01:
            probs.append(f"cell {k + 1}: tile is not one flat colour over white "
                         f"({(err[touched] > 24).mean():.3f} off)")
        cells.append(dict(label=lab, kind=kind, mask=m))
    return cells, probs


# ---------------------------------------------------------------- base icons
def core_key_of(base_icons):
    """The content key the record must carry: sha256 of the canonical JSON of the
    sorted set of base silhouettes (silhouette_class|mirrored), independent of
    roles and layout.  silhouette_class names the smallest path among icon files
    that draw the same silhouette (SILHOUETTE_CLASS); check_base_icons ties it
    to that table and to the pixels."""
    v = dict(task="find_the_shadow",
             base=sorted(f"{b['silhouette_class']}|{int(b['mirrored'])}" for b in base_icons))
    s = json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()


@lru_cache(maxsize=None)
def _library():
    'shared.iconlib.IconLibrary of this checkout and a path -> entry index.'
    import shared.iconlib as iconlib
    if not iconlib.HAVE_CAIROSVG:
        raise RuntimeError("cairosvg is needed to re-rasterise the base icons")
    lib = iconlib.IconLibrary()
    return iconlib, lib, {e["path"]: e for e in lib.icons}


@lru_cache(maxsize=4096)
def icon_mask(path, mirrored, orientation=None):
    """The named icon rasterised live from its SVG, cropped to its ink, centred on
    a square and drawn on a 92 px tile over white -> the same 50%-alpha mask
    alpha_mask() reads from the PNG.  None if the path is not in the library."""
    from PIL import Image
    _il, lib, by_path = _library()
    e = by_path.get(path)
    if e is None:
        return None
    img = lib.render(e, TILE * 4, color=(0, 0, 0))
    ys, xs = np.nonzero(np.asarray(img)[..., 3] > 8)
    if len(xs) == 0:
        return None
    crop = img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = crop.size
    side = max(w, h)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.alpha_composite(crop, ((side - w) // 2, (side - h) // 2))
    if orientation:
        T = Image.Transpose
        sq = sq.transpose((None, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_180, T.ROTATE_90, T.ROTATE_270,
                           T.TRANSPOSE, T.TRANSVERSE)[int(orientation)])
    elif mirrored:
        sq = sq.transpose(Image.FLIP_LEFT_RIGHT)
    t = sq.resize((TILE, TILE), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE), (255, 255, 255))
    canvas.paste(t, (0, 0), t)
    m, _ = alpha_mask(np.asarray(canvas))
    m.setflags(write=False)
    return m


def check_base_icons(record, meta, t, tile_masks):
    """core_key, roles and named paths against meta.cells and the pixels.
    `tile_masks` maps a 1-based slot to the mask decoded from that tile."""
    probs = []
    L = t["L"]
    bases = meta.get("base_icons") or []
    if len(bases) != L + 2:
        return [f"meta: {len(bases)} base icons, want {L + 2}"]
    law = record.get("tier") in LAW_TIERS
    if any(not isinstance(b.get("silhouette_class"), str) or not isinstance(b.get("path"), str)
           for b in bases):
        return probs + ["meta: a base icon lacks a path or a silhouette_class"]
    if law:
        lt = verify_table()
        for b in bases:
            g = b.get("orientation")
            if not isinstance(g, int) or isinstance(g, bool) or not 0 <= g < 8:
                probs.append(f"meta: {b['path']} has no orientation code 0..7")
                continue
            if b.get("mirrored") is not (g in REFLECTIONS):
                probs.append(f"meta: {b['path']} mirrored flag disagrees with orientation {g}")
            if g and b["path"] in lt["glyph"]:
                probs.append(f"meta: {b['path']} is a text glyph drawn in orientation {g}")
            want = lt["ident"].get(b["path"], b["path"])
            if b["silhouette_class"] != want:
                probs.append(f"meta: {b['path']} has identity class {b['silhouette_class']!r}, "
                             f"the release's verify table says {want!r}")
        if len({b["silhouette_class"] for b in bases}) != len(bases):
            probs.append("meta: two base icons share an identity class")
    else:
        if any(b.get("mirrored") is not False for b in bases):
            probs.append("meta: a base icon is mirrored (the release draws upright icons only)")
        # one silhouette set, one key: the class is the declared one, never a
        # self-assigned or merged class (a self class on a twin icon would give
        # the same item a second key)
        for b in bases:
            want = SILHOUETTE_CLASS.get(b["path"], b["path"])
            if b["silhouette_class"] != want:
                probs.append(f"meta: {b['path']} has silhouette_class {b['silhouette_class']!r}, "
                             f"the release's class table says {want!r}")
    _, _, by_path = _library()
    for b in bases:
        e = by_path.get(b["path"])
        if e is not None and (e.get("split") != "train" or e.get("style") != "silhouette"):
            probs.append(f"meta: {b['path']} is a {e.get('split')}-split {e.get('style')} icon; "
                         f"the release draws train-split silhouette icons only")
    if record.get("core_key") != (core_key_of_law(bases) if law else core_key_of(bases)):
        probs.append("core_key is not the key of meta.base_icons")
    cells = meta.get("cells") or []
    for c in cells:
        if c.get("kind") == "pattern" and bases[c["base"]].get("role") != f"pattern {c['label']}":
            probs.append(f"meta: base icon {c['base']} role {bases[c['base']].get('role')!r} "
                         f"but it is drawn as pattern {c['label']}")
    if bases[L].get("role") != ROLE_U or bases[L + 1].get("role") != ROLE_LONE:
        probs.append("meta: base icons L and L+1 are not the answerless family and the lone shadow")
    if probs:
        return probs
    shows = [(c["base"], c["slot"]) for c in cells if c["kind"] == "pattern"]
    shows += [(c["shadow"], c["slot"]) for c in cells if c["kind"] == "shadow" and not c.get("clone")]
    for b in bases:
        if b["silhouette_class"] == b["path"] or law:
            # law tiers: identity classes join icons through ANY orientation; they are pinned by the
            # verify table (VERIFY_TABLE_SHA256), not re-derived per item
            continue
        r, c = icon_mask(b["path"], bool(b["mirrored"])), icon_mask(b["silhouette_class"], bool(b["mirrored"]))
        if r is None or c is None:
            probs.append(f"meta: {b['path']!r} or its class {b['silhouette_class']!r} is not in the icon library")
        elif iou(r, c) < TWIN_IOU_MIN or disc(r, c) > TWIN_DISC_MAX:
            probs.append(f"meta: {b['path']} is not a silhouette twin of its class {b['silhouette_class']} "
                         f"(IoU {iou(r, c):.3f}, {disc(r, c):.1f} px)")
    for k, slot in shows:
        b = bases[k]
        r = icon_mask(b["path"], bool(b["mirrored"]), b.get("orientation") if law else None)
        if r is None:
            probs.append(f"meta: base icon {b['path']!r} is not in the icon library")
            continue
        m = tile_masks[slot]
        if iou(r, m) < BASE_IOU_MIN or disc(r, m) > BASE_DISC_MAX:
            probs.append(f"cell {slot}: does not show the named icon {b['path']} "
                         f"(IoU {iou(r, m):.3f}, {disc(r, m):.1f} px)")
    return probs


# ---------------------------------------------------------------- meta gold
def meta_gold(meta, t):
    """Gold string from the construction record alone, plus structural checks."""
    probs = []
    L, N, C, UC = t["L"], t["N"], t["C"], t["UC"]
    cells = meta.get("cells") or []
    if len(cells) != t["rows"] * t["cols"]:
        return None, [f"meta lists {len(cells)} cells, grid has {t['rows'] * t['cols']}"]
    if [c["slot"] for c in cells] != list(range(1, len(cells) + 1)):
        probs.append("meta cells not in reading order")
    pats = [c for c in cells if c["kind"] == "pattern"]
    shads = [c for c in cells if c["kind"] == "shadow"]
    if [c["label"] for c in pats] != [chr(ord("A") + i) for i in range(L)]:
        probs.append("meta: patterns are not lettered A.. in reading order")
    if [c["label"] for c in shads] != [str(i) for i in range(1, N + 1)]:
        probs.append("meta: shadows are not numbered 1.. in reading order")
    if sorted(c["base"] for c in pats) != list(range(L)):
        probs.append("meta: pattern bases are not 0..L-1")
    if sorted(c["shadow"] for c in shads) != list(range(N)):
        probs.append("meta: shadow indices are not 0..N-1")
    real = {c["shadow"]: c for c in shads if not c.get("clone")}
    if sorted(real) != list(range(L + 2)):
        probs.append(f"meta: unedited shadows are {sorted(real)}, want 0..{L + 1}")
    fam = {}
    for c in shads:
        if c.get("clone"):
            fam[c["family_base"]] = fam.get(c["family_base"], 0) + 1
            if c.get("edit_kind") not in ("bite", "fill") or c.get("polarity") not in ("smoother", "extra"):
                probs.append(f"meta: shadow {c['label']} has no valid edit record")
    want = {k: C for k in range(L)}
    want[L] = UC
    if fam != want:
        probs.append(f"meta: clone families {fam} != {want}")
    bases = meta.get("base_icons") or []
    if len(bases) != L + 2 or len({(b["path"], b["mirrored"]) for b in bases}) != L + 2:
        probs.append("meta: base icons are not L+2 distinct icons")
    if probs:
        return None, probs
    pairs = {}
    for c in pats:
        pairs[c["label"]] = int(real[c["base"]]["label"])
    return format_answer(pairs, N), probs


# ---------------------------------------------------------------- verify
def verify(record, image) -> list[str]:
    probs = []
    tier = record.get("tier")
    if record.get("task") != "find_the_shadow" or tier not in TIERS:
        return [f"not a find_the_shadow record of a known tier: {record.get('task')}/{tier}"]
    t = tier_numbers(tier)
    L, N, U = t["L"], t["N"], t["U"]
    if record.get("answer_type") != "text" or record.get("choices") is not None:
        probs.append("answer_type must be text with no choices")
    q = record.get("question", "")
    if q != expected_question(tier):
        probs.append("question text differs from the declared form for this tier")
    if not q.isascii() or not record.get("answer", "").isascii():
        probs.append("non-ASCII question or answer")
    try:
        gold_pairs, _gold_um = parse_answer(record.get("answer", ""), L, N, U)
    except ValueError as exc:
        return probs + [str(exc)]
    # the format example must be impossible as an answer: no shared letter,
    # number, pair, first pair or first unmatched number with the gold
    ex = question_example(q)
    if ex is None:
        probs.append("question carries no format example in the declared form")
    else:
        try:
            ov = example_overlap(ex, record["answer"])
        except ValueError as exc:
            ov = [str(exc)]
        if ov:
            probs.append(f"the question's format example {ex!r} shares {'; '.join(ov)} with the gold")

    meta = record.get("meta") or {}
    if (meta.get("canvas") != [t["W"], t["H"]] or meta.get("grid") != [t["rows"], t["cols"]]
            or meta.get("letters") != L or meta.get("shadows") != N or meta.get("unmatched") != U
            or meta.get("tile_px") != TILE):
        probs.append("meta canvas/grid/counts disagree with the tier")
    mg, mp = meta_gold(meta, t)
    probs += mp
    if mg is not None and mg != record["answer"]:
        probs.append(f"answer from meta {mg} != recorded answer")

    # ---------------------------------------------- the PNG alone
    cells, dp = decode(image, tier)
    probs += dp
    if cells is None:
        return probs
    pat = {c["label"]: c for c in cells if c["kind"] == "pattern"}
    sha = {c["label"]: c for c in cells if c["kind"] == "shadow"}
    if sorted(pat) != [chr(ord("A") + i) for i in range(L)]:
        probs.append(f"pixels: lettered colour cells are {sorted(pat)}")
    if sorted(sha, key=lambda s: (len(s), s)) != [str(i) for i in range(1, N + 1)]:
        probs.append(f"pixels: numbered dark cells are {sorted(sha, key=lambda s: (len(s), s))}")
    for c in cells:
        if c["kind"] == "pattern" and not c["label"].isalpha():
            probs.append(f"pixels: colour pattern labelled {c['label']}")
        if c["kind"] == "shadow" and not c["label"].isdigit():
            probs.append(f"pixels: shadow labelled {c['label']}")
    mcells = meta.get("cells") or []
    if len(mcells) == len(cells):
        for k, (mc, c) in enumerate(zip(mcells, cells)):
            if mc.get("label") != c["label"] or mc.get("kind") != c["kind"]:
                probs.append(f"cell {k + 1}: pixels read {c['kind']} {c['label']!r}, meta says "
                             f"{mc.get('kind')} {mc.get('label')!r}")
                break
    if probs:
        return probs
    # core_key, roles and the named icons, tied to the pixels
    probs += check_base_icons(record, meta, t, {k + 1: c["mask"] for k, c in enumerate(cells)})
    if probs:
        return probs

    letters = [chr(ord("A") + i) for i in range(L)]
    numbers = [str(j) for j in range(1, N + 1)]
    P = [pat[a]["mask"] for a in letters]
    S = [sha[n]["mask"] for n in numbers]
    D = np.array([[disc(p, s) for s in S] for p in P])
    I = np.array([[iou(p, s) for s in S] for p in P])
    refuted = (I < IOU_FAR) | (D >= WRONG_MIN)
    valid = D <= TRUE_MAX
    amb = ~refuted & ~valid
    if amb.any():
        i, j = map(int, np.argwhere(amb)[0])
        probs.append(f"{letters[i]} vs shadow {numbers[j]}: departs by {D[i, j]:.1f} px "
                     f"(IoU {I[i, j]:.3f}) -- neither identical nor legibly different")
    pred = {}
    for i, a in enumerate(letters):
        js = np.flatnonzero(valid[i] & ~refuted[i])
        if len(js) != 1:
            probs.append(f"{a}: {len(js)} shadows depart from it by <= {TRUE_MAX} px")
            continue
        pred[a] = int(js[0]) + 1
    if len(set(pred.values())) != len(pred):
        probs.append("two patterns claim the same shadow")
    # "departs nowhere": the thin XOR of a true pair must hug the pattern's
    # outline -- a detached speck of ink or a 1-2 px bump / notch must not pass
    for i, a in enumerate(letters):
        if a in pred:
            o = off_outline(P[i], S[pred[a] - 1])
            if o > OUTLINE_BAND:
                probs.append(f"{a} vs shadow {pred[a]}: departs {o:.1f} px off the pattern's outline "
                             f"(more than antialiasing)")
    if probs:
        return probs
    derived = format_answer(pred, N)
    if derived != record["answer"]:
        probs.append(f"answer read from the image {derived} != recorded answer")

    # no two shadows indistinguishable
    for x in range(N):
        for y in range(x + 1, N):
            if iou(S[x], S[y]) >= IOU_FAR and disc(S[x], S[y]) < WRONG_MIN:
                probs.append(f"shadows {x + 1} and {y + 1} differ by under {WRONG_MIN} px")
    # no other shadow is a rotated / mirrored copy of a pattern
    for i, a in enumerate(letters):
        for tm in dihedral(P[i]):
            for j in range(N):
                if j + 1 == pred[a]:
                    continue
                if iou(tm, S[j]) >= IOU_FAR and disc(tm, S[j]) <= TWIN_MAX:
                    probs.append(f"shadow {j + 1} is pattern {a} rotated or mirrored")
    return probs
