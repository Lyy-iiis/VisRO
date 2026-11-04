"""find_the_shadow -- the MANY-STRONG-CLONES ladder.

WHY NEITHER SIMILARITY NOR LETTER COUNT IS THE DIAL
---------------------------------------------------
Rival similarity is not a usable dial: every rung must keep the same
refutability floor (a solid disc of radius >= 8 px inside the XOR between a
pattern and any WRONG shadow), and that floor pins the ABSOLUTE size of the one
feature a reader has to find.  Making the silhouettes globally more alike leaves
that feature exactly as big as it was.

The COUNT of letters L is a real axis but a short one: the answer is
all-or-nothing over L independent pair decisions (the unmatched list is a
FUNCTION of the pairing, not extra information), so accuracy = p^L with p close
to 1, and L is capped by pixels: with one clone per letter and one answerless
twin the grid needs >= 3L + 3 cells, and a legible 92 px tile fits 35 cells,
i.e. L <= 10.

WHAT THIS LADDER TURNS: c = STRONG CLONES PER LETTER
----------------------------------------------------
p, not L.  And it is pushed down by ADDING EVIDENCE TO READ, never by shrinking
any single piece of evidence.  Each letter has c strong clones, all in the same
0.86-0.94 IoU band, all carrying a >= 8 px identifiable disc against their
parent, and any two of them >= 8 px apart from EACH OTHER as well.  The right
shadow still differs from its pattern by <= 1.0 px -- antialiasing only.  So the
call is "in a family of c+1 nearly identical silhouettes, find the one carrying
nothing at all", which is c comparisons instead of one, each of them equally
legible.  With p(c) ~ p1^c the all-or-nothing accuracy is p1^(L*c), and the
quantity to maximise under the pixel budget is L*c, not L.

L IS TRADED FOR c
-----------------
Cells needed = L (patterns) + L(c+1) (each letter and its clones) + the
answerless family + one lone shadow.  With a 92 px tile and 451,584 px the grid
holds 35 cells, so L <= (32 - c)/(c + 2) ... and L*c still rises with c:

    c=1  L=8   L*c=10.7 (one clone/letter)   c=4  L=5   L*c=20
    c=2  L=8   L*c=16                        c=5  L=4   L*c=20
    c=3  L=6   L*c=18                        c=6  L=4   L*c=24

The top three rungs take c = 2 / 3 / 4 at L = 6 / 6 / 5, i.e. L*c = 12 / 18 / 20.
c = 6 is not reachable: on a 92 px tile the icon pool cannot carry six mutually
separable blemishes on one contour, so L*c = 20 is the ceiling of this axis
under the 576-visual-token budget.

WHAT MUST NOT MOVE (else the task degenerates into guessing)
------------------------------------------------------------
  * true pair differs by <= 1.0 px  (pure antialiasing)          -- tightened
  * every clone is refuted by a >= 8.0 px inscribed disc          -- unchanged
  * every clone-vs-clone pair is ALSO >= 8.0 px apart             -- gate kept
                                        (no two shadows are indistinguishable)
  * polarity is mixed INSIDE every family: at least one clone is the plainer of
    the pair and at least one carries an extra lobe, so "the one with the odd
    bump is fake" is not even a per-letter strategy.
  * a full answerless look-alike family plus one lone shadow, so "every family
    donates one answer" is false.
  * gate tables are built on the DELIVERED mask, and each tile's silhouette
    is cut at 50% of its own ink.

CANVASES (all inside 576 visual tokens, w%28 == h%28 == 0)
-----------------------------------------------------------
  easy    644x364  3x6 = 18 cells  tile 92  234,416 px  (299 tokens)
  medium  728x364  3x7 = 21 cells  tile 92  264,992 px  (338 tokens)
  hard    728x476  4x7 = 28 cells  tile 92  346,528 px  (442 tokens)
  xhard   728x616  5x7 = 35 cells  tile 92  448,448 px  (572 tokens)
  xxhard  728x616  5x7 = 35 cells  tile 92  448,448 px  (572 tokens)
The tile is 92 px on ALL FIVE rungs, so per-judgment legibility is literally
identical across the ladder and only the NUMBER of judgments moves.
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

from . import decoys
from .base import silhouette

SS = 4
PIXEL_CAP = 451584
PATCH = 28
MARGIN, GAP, LABEL_H = 5, 4, 18
R_TARGET = 12.0                 # designed blemish radius, px, SAME on every rung

# Everything that sets the difficulty of ONE pattern-vs-shadow judgment is the
# same on all three rungs: same 92 px tile, same 8 px refutation floor, same 12 px
# designed blemish, same natural-rival band, same clone band.  Only c moves.
# `rad_pre` and `fill_min` are raised, and `mult_thr` lowered, for the clones:
# With c clones per family the number of CROSS-family confusable pairs grows like
# c^2 (a clone of X against every clone of Y), and each one still has to be
# refutable by an 8 px disc: at rad_pre 9.5 that gate rejected 86% of otherwise
# good images at c = 4.  rad_pre 11.0 gives the base pair 1.5 px of headroom for
# the blemishes to eat, fill_min 0.36 makes the silhouettes big enough that a
# blemish inside the 0.86-0.945 IoU band is fat in absolute pixels, and mult_thr
# 0.72 pays for both by tripling the supply of natural-rival edges.  None of this
# touches the shipped floors: the clone band, the 8 px refutation disc, the 1 px
# true-pair flatness and the polarity mixing are the same on all three rungs.
BASE = dict(hi_nat=0.855, mult_thr=0.72, rmin=8.0, rmax=21.0, sb_max=2,
            out_max=0.97, clone_lo=0.86, clone_hi=0.945, fill_min=0.36,
            rad_pre=11.0, natural="on", free_pair_max=0.62)

# ---------------------------------------------------------------- the two LOW
# rungs, so that ONE table holds all five:
#
#     easy    L = 5   natural = "off"
#     medium  L = 6   natural = "off"
#
# Shadows keep the pattern's orientation and scale on every rung.  `natural`
# says whether the icons of one image are chosen to look alike: "off" picks the
# base icons freely subject to pairwise IoU < free_pair_max = 0.62
# (select_base_free), "on" is select_base, which deliberately picks DISJOINT
# NATURAL-RIVAL EDGES so that beyond its manufactured clones every letter also
# faces a real icon in the 0.72-0.855 band.
#
# The unified question always asks for the pairing AND the complete unmatched
# list, and tells the reader that the shadows come in look-alike families.  The
# minimum structure that keeps it literally true is c = 1 clone per letter plus
# the answerless family and the lone shadow (NU = 2, UC = [1, 0]), i.e. 3L + 3
# cells: the one-clone geometry, one rung below "hard" on this file's c ladder.
# 3L + 3 = 18 / 21 cells do not fit in 448x448 at a 92 px tile (12 cells is the
# most it holds), and the 92 px tile is what makes ONE pattern-vs-shadow
# judgment equally legible on all five rungs, so the tile is kept and the
# canvas grown; both low rungs are still smaller than hard.
TIERS = {
    # 3x6 = 18 cells, 234,416 px (299 visual tokens)
    "easy": dict(BASE, W=644, H=364, COLS=6, ROWS=3, TILE=92, L=5, C=1,
                 NU=2, UC=[1, 0], natural="off"),
    # 3x7 = 21 cells, 264,992 px (338 visual tokens)
    "medium": dict(BASE, W=728, H=364, COLS=7, ROWS=3, TILE=92, L=6, C=1,
                   NU=2, UC=[1, 0], natural="off"),
    # 4x7 = 28 cells, 346,528 px (442 visual tokens)
    "hard": dict(BASE, W=728, H=476, COLS=7, ROWS=4, TILE=92, L=6, C=2,
                 NU=2, UC=[2, 0]),
    # 5x7 = 35 cells, 448,448 px (572 visual tokens)
    "xhard": dict(BASE, W=728, H=616, COLS=7, ROWS=5, TILE=92, L=6, C=3,
                  NU=2, UC=[3, 0]),
    # 5x7 = 35 cells, 448,448 px (572 visual tokens)
    "xxhard": dict(BASE, W=728, H=616, COLS=7, ROWS=5, TILE=92, L=5, C=4,
                   NU=2, UC=[3, 0]),
}
for _t, _d in TIERS.items():
    _d["letter_clones"] = [_d["C"]] * _d["L"]
    _d["u_clones"] = list(_d["UC"])
    _d["NSHAD"] = _d["L"] + _d["NU"] + sum(_d["letter_clones"]) + sum(_d["u_clones"])
    _d["NCLONE"] = sum(_d["letter_clones"]) + sum(_d["u_clones"])
    # sites for c clones have to fit on one contour without piling up
    _d["avoid_frac"] = round(float(min(0.30, 0.56 / np.sqrt(_d["C"] + 1))), 3)
    assert _d["L"] + _d["NSHAD"] == _d["ROWS"] * _d["COLS"], (_t, _d["NSHAD"])
    # every letter carries exactly c clones.  c >= 2 on the three shipped rungs;
    # c = 1 is allowed only for the low rungs (one-clone geometry), where the
    # per-family polarity gate below is undefined and the image-wide one carries
    # the load instead.
    assert min(_d["letter_clones"]) == _d["C"] >= 1, _t
    assert _d["C"] >= 2 or _d["natural"] == "off", _t
    assert _d["natural"] in ("on", "off"), _t
    assert _d["u_clones"][0] >= 1 and _d["u_clones"][-1] == 0, _t
    assert len(_d["u_clones"]) == _d["NU"], _t   # one answerless family + a loner
    assert _d["W"] % PATCH == 0 and _d["H"] % PATCH == 0
    assert _d["W"] * _d["H"] <= PIXEL_CAP, (_t, _d["W"] * _d["H"])

ACCEPT = dict(self_iou_min=0.965, disc_min=8.0, true_disc_max=1.0,
              xor_ratio_min=2.5, outline_frac_min=0.60)

# The question states the matching rules once each; its format example uses letters (Y, Z) and
# numbers (91-94) that no rung can produce, so it shares nothing with any gold.
QUESTION = (
    'Each colored pattern A-{last} has exactly one same-shape dark shadow among 1-{N}, none '
    'shared, so {U} match nothing. Shadows are never rotated, mirrored or rescaled; holes count. '
    'Look-alikes differ by one small edge edit; the match differs nowhere. Answer pairs in letter'
    " order, then ';unmatched:' and the rest ascending, e.g. Y-93,Z-91;unmatched:92,94."
)


# ------------------------------------------------------------------ helpers
def canon_tile(img, size):
    a = np.asarray(img.convert("RGBA"))
    ys, xs = np.where(a[..., 3] > 8)
    if len(xs) == 0:
        return img.resize((size, size), Image.LANCZOS)
    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    m = max(w, h)
    sq = Image.new("RGBA", (m, m), (0, 0, 0, 0))
    sq.alpha_composite(crop, ((m - w) // 2, (m - h) // 2))
    return sq.resize((size, size), Image.LANCZOS)


def shadow_rgba(mask, color):
    """Rasterise a boolean silhouette to a solid single-colour RGBA tile.  TRUE
    shadows and MANUFACTURED shadows go through this same function, so a clone
    carries no rendering fingerprint that would give it away."""
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.9)
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


_T = Image.Transpose
# orientation codes 0..7 of a base icon: the 8 symmetries of the square tile
ORIENT_OPS = (None, _T.FLIP_LEFT_RIGHT, _T.FLIP_TOP_BOTTOM, _T.ROTATE_180, _T.ROTATE_90, _T.ROTATE_270,
              _T.TRANSPOSE, _T.TRANSVERSE)


def base_tile(lib, ents, ref, S, tint):
    """One base icon at the delivery size.  `ref` is (pool index, orientation
    code 0..7 of ORIENT_OPS); a bool reads as 0 / 1, i.e. upright / mirrored.

    An icon and the same icon SEEN IN A MIRROR (or rotated) are two different
    silhouettes, and the pool is small enough that this matters: the
    natural-rival graph the hard rungs need has 103 edges over 1267 upright
    icons, and it grows like the SQUARE of the pool.  The orientation is applied
    to the finished square tile, so the pattern and the shadow built from it are
    transformed by exactly the same operation and carry no rendering
    fingerprint that would give it away.  Every gate then sees an ordinary
    silhouette; the cache tables of the other orientations are rendered this way
    rather than transformed from the stored masks, because canon_tile's centring
    is not symmetric on an odd offset."""
    k, g = ref
    img = canon_tile(silhouette.render_canon(lib, ents[k], S, tint), S)
    op = ORIENT_OPS[int(g)]
    return img if op is None else img.transpose(op)


# ------------------------------------------------------------------ selection
def select_tables(M, RAD, OUT, avail, dp):
    """The gate masks, built once per rung.  M / OUT / RAD are tabulated on the
    masks the shipped tile actually draws, at the shipped tile size, so `rmin` is
    the delivered pixel count with no conversion."""
    thr = dp["rad_pre"]
    OK = (M < 0.62) | (RAD >= thr)
    ALLOW = OK & (M <= dp["hi_nat"]) & ~((M >= 0.80) & (OUT > dp["out_max"])) \
        & avail[None, :]
    np.fill_diagonal(ALLOW, False)
    SB = (OUT >= 0.93).copy(); np.fill_diagonal(SB, False)
    BAND = ALLOW & (M >= dp["mult_thr"])
    E = np.argwhere(np.triu(BAND, 1))
    return ALLOW, SB, BAND, E, float(thr)


def select_base(M, T, avail, nbase, dp, rng, tries=400):
    """Pick the NBASE real icons as a set of DISJOINT natural-rival edges, so
    besides its c manufactured clones every letter ALSO has a real icon nearby."""
    ALLOW, SB, BAND, E, _ = T
    n = ALLOW.shape[0]
    for _ in range(tries):
        feas = avail.copy()
        sbc = np.zeros(n, np.int32)
        chosen = []
        while len(chosen) + 2 <= nbase:
            m = feas[E[:, 0]] & feas[E[:, 1]]
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                break
            a, b = (int(x) for x in E[idx[rng.randrange(len(idx))]])
            chosen += [a, b]
            feas &= ALLOW[a] & ALLOW[b]
            sbc = sbc + SB[a] + SB[b]
            feas &= sbc <= dp["sb_max"]
            feas[[a, b]] = False
        if len(chosen) == nbase - 1:                    # odd nbase: one extra
            c = np.flatnonzero(feas & BAND[chosen].any(0))
            if len(c) == 0:
                continue
            chosen.append(int(c[rng.randrange(len(c))]))
        if len(chosen) != nbase:
            continue
        rng.shuffle(chosen)
        return chosen


def select_base_free(M, T, avail, nbase, dp, rng, tries=400):
    """The `natural = "off"` regime, for the easy and medium rungs.

    Icons are sampled FREELY and a set is only rejected when a pairwise IoU
    reaches free_pair_max = 0.62 (silhouettes must be clearly different), on the
    same 92 px delivered-mask IoU table `M` that every other gate here uses, so
    on these two rungs a letter's ONLY confusable neighbours are its own c
    manufactured clones -- there is no natural rival.

    Everything else -- ALLOW (refutability of every kept pair) and the sb_max
    same-outline budget -- is select_base's, unchanged."""
    ALLOW, SB, BAND, E, _ = T
    n = ALLOW.shape[0]
    cap = dp["free_pair_max"]
    for _ in range(tries):
        feas = avail.copy()
        sbc = np.zeros(n, np.int32)
        chosen = []
        while len(chosen) < nbase:
            idx = np.flatnonzero(feas)
            if len(idx) == 0:
                break
            a = int(idx[rng.randrange(len(idx))])
            chosen.append(a)
            feas &= ALLOW[a] & (M[a] < cap) & (M[:, a] < cap)
            sbc = sbc + SB[a]
            feas &= sbc <= dp["sb_max"]
            feas[a] = False
        if len(chosen) != nbase:
            continue
        rng.shuffle(chosen)
        return chosen


# ------------------------------------------------------------------ build one
def tally(why, k):
    if why is not None:
        why[k] = why.get(k, 0) + 1


def polarity_plan(c, rng):
    """Polarity assignment INSIDE one family.  With c >= 2 both polarities are
    forced to appear, so within a single letter the reader cannot decide by
    "the plain one is real" or "the bumpy one is real": it has to compare each
    candidate against the coloured pattern."""
    if c == 1:
        return [rng.choice(["smoother", "extra"])]
    pols = ["smoother", "extra"] + [rng.choice(["smoother", "extra"])
                                    for _ in range(c - 2)]
    rng.shuffle(pols)
    return pols


def build_shadows(lib, ents, idxs, dp, rng, why=None):
    """The NBASE real silhouettes + c manufactured near-clones for every letter
    (and for the answerless family).

    The blemish is sized to a FIXED radius (R_TARGET px on the delivered tile) by
    turning it into a per-silhouette IoU target, 1 - pi r^2 / area, and then
    clipped into the clone band, so a clone is never a giveaway and never a
    coin flip.  Sites inside one family are kept apart by `avoid_frac`, which
    shrinks as c grows -- the >= 8 px clone-vs-clone gate on the rendered PNG is
    what actually decides whether two clones stayed distinguishable."""
    L, S, TILE = dp["L"], dp["TILE"] * SS, dp["TILE"]
    base_masks, tgt = [], []
    for k in idxs:                      # k is (pool index, orientation)
        img = base_tile(lib, ents, k, S, (0, 0, 0))
        m = np.asarray(img)[..., 3] > 127   # the icon's own 50% contour, which is
        #                                     also where ink_mask cuts the tile
        base_masks.append(m)
        area = float(decoys.norm_mask(m, TILE).sum())
        tgt.append(float(np.clip(1.0 - np.pi * R_TARGET ** 2 / max(area, 1.0),
                                 dp["clone_lo"] + 0.015, dp["clone_hi"] - 0.005)))
    other = {"smoother": "extra", "extra": "smoother"}

    plan = [(i, c) for i, c in enumerate(dp["letter_clones"])] + \
           [(L + i, c) for i, c in enumerate(dp["u_clones"])]
    masks = list(base_masks)
    meta = [dict(kind="real", src=j) for j in range(len(base_masks))]
    order = [(b, c) for b, c in plan if c]
    rng.shuffle(order)
    for b, c in order:
        pols = polarity_plan(c, rng)
        used, fam_n, blocked = [], [], []
        for ci in range(c):
            want = pols[ci]
            got = None
            # the first two clones of a family MUST take the two polarities (that
            # is what forbids "the plain one is the real one" per letter); later
            # ones may fall back, the mix is already guaranteed.
            cands = ([want] * 6 if ci < 2 else
                     [want, want, want, other[want], other[want], other[want]])
            for att, pp in enumerate(cands):
                shrink = (1.0, 1.0, 0.86, 0.86, 0.72, 0.62)[min(att, 5)]
                cand = decoys.make_decoy(
                    base_masks[b], tgt[b], TILE, dp["rmin"], dp["rmax"], rng,
                    polarity=pp, why=why, avoid=used + blocked,
                    avoid_r=dp["avoid_frac"] * shrink * S)
                if cand is None:
                    continue
                # ---- CLONES MUST ALSO BE >= rmin APART FROM EACH OTHER.  A
                # candidate whose blemish sits too near an accepted one is thrown
                # back HERE, and its site is blocked, instead of costing the whole
                # image a re-roll; this is what makes c = 6 affordable at all.
                cn = decoys.norm_mask(cand[0], TILE)
                if any(decoys.disc_radius(cn, f) < dp["rmin"] + 0.5 for f in fam_n):
                    blocked.append(tuple(cand[1]["site"]))
                    tally(why, "clone_too_close")
                    continue
                got = cand; pols[ci] = pp
                break
            if got is None:
                tally(why, "fail_base")
                return None
            m2, info = got
            used.append(tuple(info["site"]))
            fam_n.append(decoys.norm_mask(m2, TILE))
            masks.append(m2)
            meta.append(dict(kind="clone", src=b, family=b,
                             target_iou=round(tgt[b], 4), **info))
    return masks, meta


def render(lib, ents, idxs, masks, dp, rng):
    L, NSHAD, TILE = dp["L"], dp["NSHAD"], dp["TILE"]
    ROWS, COLS, W, H = dp["ROWS"], dp["COLS"], dp["W"], dp["H"]
    S = TILE * SS
    tints = rng.sample(silhouette.SPECIFIC_TINTS, L) if L <= len(silhouette.SPECIFIC_TINTS) \
        else [rng.choice(silhouette.SPECIFIC_TINTS) for _ in range(L)]
    shadow_col = rng.choice(silhouette.SHADOW_COLORS)
    spec = [base_tile(lib, ents, idxs[k], S, tints[k]) for k in range(L)]
    shad = [shadow_rgba(m, shadow_col) for m in masks]

    cw = (W - 2 * MARGIN - (COLS - 1) * GAP) // COLS
    ch = (H - 2 * MARGIN - (ROWS - 1) * GAP) // ROWS
    assert TILE + 6 <= cw and TILE + LABEL_H + 3 <= ch, (cw, ch, TILE)
    gw = COLS * cw + (COLS - 1) * GAP
    gh = ROWS * ch + (ROWS - 1) * GAP
    ox, oy = (W - gw) // 2, (H - gh) // 2
    tx = (cw - TILE) // 2

    content = [("p", k) for k in range(L)] + [("s", j) for j in range(NSHAD)]

    def rc(slot):
        return divmod(slot - 1, COLS)

    ok = False
    for _ in range(4000):                 # pattern and its shadow never adjacent
        order = list(range(len(content)))
        rng.shuffle(order)
        slot_of = {content[i]: order[i] + 1 for i in range(len(content))}
        ok = True
        for k in range(L):
            (r1, c1), (r2, c2) = rc(slot_of[("p", k)]), rc(slot_of[("s", k)])
            if max(abs(r1 - r2), abs(c1 - c2)) < 2:
                ok = False; break
        if ok:
            break
    if not ok:
        return None
    slot_content = {slot_of[c]: c for c in content}

    img = Image.new("RGB", (W, H), silhouette.BG)
    d = ImageDraw.Draw(img)
    font = load_bundled_font(15, "sans-bold")
    boxes, tboxes, letter_of, number_of = {}, {}, {}, {}
    li = ni = 0
    slot = 1
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = ox + c * (cw + GAP), oy + r * (ch + GAP)
            boxes[slot] = (x0, y0, x0 + cw, y0 + ch)
            d.rectangle(boxes[slot], outline=(212, 212, 212), width=2)
            tb = (x0 + tx, y0 + LABEL_H, x0 + tx + TILE, y0 + LABEL_H + TILE)
            tboxes[slot] = tb
            kind, k = slot_content[slot]
            silhouette.fit_tile(img, tb, spec[k] if kind == "p" else shad[k])
            if kind == "p":
                lab = chr(ord("A") + li); li += 1; letter_of[k] = lab
            else:
                ni += 1; lab = str(ni); number_of[k] = lab
            d.text((x0 + 6, y0 + 1), lab, fill=(15, 15, 15), font=font)
            slot += 1
    pairs = [(slot_of[("p", k)], slot_of[("s", k)]) for k in range(L)]
    unmatched = sorted(int(number_of[j]) for j in range(L, NSHAD))
    answer = (",".join(f"{letter_of[k]}-{number_of[k]}"
                       for k in sorted(range(L), key=lambda k: letter_of[k]))
              + ";unmatched:" + ",".join(str(x) for x in unmatched))
    return img, boxes, tboxes, slot_content, pairs, answer, letter_of, number_of


# ------------------------------------------------------------------ PNG audit
def ink_mask(gray_tile):
    """Foreground of one tile, thresholded at HALF of that tile's own ink density.
    A fixed cut is not the same cut for a pale pattern as for a near-black shadow;
    cutting each tile at 50% of its own ink is the same geometric level in both."""
    lo = float(np.percentile(gray_tile, 1.0))
    return gray_tile < (255.0 + lo) / 2.0


def png_masks(img, tboxes, TILE):
    a = np.asarray(img.convert("L"))
    out = {}
    for slot, (x0, y0, x1, y1) in tboxes.items():
        sub = ink_mask(a[y0:y1, x0:x1])
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            sub = lab == k
        out[slot] = decoys.norm_mask(sub, TILE)
    return out


def best_assignment(Sm):
    Lr, N = Sm.shape
    r, c = linear_sum_assignment(Sm, maximize=True)
    perm = [int(c[list(r).index(i)]) for i in range(Lr)]
    val = float(sum(Sm[i, perm[i]] for i in range(Lr)))
    second = -1e9
    for i in range(Lr):
        S2 = Sm.copy(); S2[i, perm[i]] = -1e6
        r2, c2 = linear_sum_assignment(S2, maximize=True)
        second = max(second, float(sum(Sm[int(a), int(b)] for a, b in zip(r2, c2))))
    return perm, val, second


def audit(img, tboxes, slot_content, dp, meta=None):
    """Everything re-derived from the RENDERED PNG."""
    L, NSHAD, TILE, C = dp["L"], dp["NSHAD"], dp["TILE"], dp["C"]
    masks = png_masks(img, tboxes, TILE)
    pslot = {k: s for s, (t, k) in slot_content.items() if t == "p"}
    sslot = {j: s for s, (t, j) in slot_content.items() if t == "s"}
    A = [masks[pslot[i]] for i in range(L)]
    B = [masks[sslot[j]] for j in range(NSHAD)]

    best = np.zeros((L, NSHAD)); disc = np.full((L, NSHAD), 99.0)
    xorA = np.zeros((L, NSHAD)); ofrac = np.zeros((L, NSHAD))
    OA = [decoys.outline_fill(a) for a in A]
    OB = [decoys.outline_fill(b) for b in B]
    for i in range(L):
        for j in range(NSHAD):
            best[i, j] = decoys.iou(A[i], B[j])
            x = np.logical_xor(A[i], B[j])
            xorA[i, j] = float(x.sum())
            if best[i, j] >= 0.62:
                disc[i, j] = decoys.disc_radius(A[i], B[j])
                ofrac[i, j] = (float(np.logical_xor(OA[i], OB[j]).sum())
                               / max(xorA[i, j], 1.0))
    # ---- shadow vs shadow: no two shadows anywhere may be indistinguishable
    ss_disc, ss_pairs, fam_disc = [], 0, []
    fam = {}
    if meta is not None:
        for j, m in enumerate(meta):
            f = m["src"] if m["kind"] == "clone" else j
            fam.setdefault(f, []).append(j)
    for i in range(NSHAD):
        for j in range(i + 1, NSHAD):
            if decoys.iou(B[i], B[j]) >= 0.62:
                ss_pairs += 1
                d = decoys.disc_radius(B[i], B[j])
                ss_disc.append(d)
    for f, mem in fam.items():
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                fam_disc.append(decoys.disc_radius(B[mem[x]], B[mem[y]]))
    # ---- THE "never look at the pattern" SHORTCUT, measured on the PNG.
    # Inside one look-alike family the true shadow and its clones differ by a
    # notch (clone loses ink) or a lobe (clone gains ink).  If the same sign
    # held for every letter, "the plainest / the fattest member of each family
    # is the real one" would answer the whole image without ever looking at a
    # coloured pattern.  area_rule_hits is the better of those two global rules,
    # in letters solved; the adapter's gate requires it to be < L.  Mixing the
    # polarity across the image is what makes it fail, and that is why the
    # mixing gate matters most on the c = 1 rungs, where one family cannot hold
    # both polarities by itself.
    area_rule = 0
    if meta is not None:
        ar = [float(b.sum()) for b in B]
        for pick in (min, max):
            area_rule = max(area_rule, sum(
                1 for i in range(L)
                if pick(fam.get(i, [i]), key=lambda j: ar[j]) == i))

    diag = best[np.arange(L), np.arange(L)]
    off = best.copy(); off[np.arange(L), np.arange(L)] = -1.0
    srt = np.sort(off, 1)[:, ::-1]
    row_riv, row_2nd = srt[:, 0], srt[:, 1]
    riv_c = srt[:, C - 1]                 # the c-th strongest rival of each letter
    riv_c1 = srt[:, C]                    # the first rival BEYOND the clone family
    riv_j = off.argmax(1)
    topc_idx = [list(np.argsort(-off[i])[:C]) for i in range(L)]
    col_riv = np.array([max([best[x, k] for x in range(L) if x != k])
                        for k in range(L)])
    uniq_row = bool((diag > row_riv).all())
    uniq = bool(uniq_row and (diag > col_riv).all())
    opt, val, sec = best_assignment(best)
    exh = bool(tuple(opt) == tuple(range(L)) and val - sec > 1e-9)

    true_disc = [decoys.disc_radius(A[i], B[i]) for i in range(L)]
    xor_true = np.array([xorA[i, i] for i in range(L)])
    xor_riv = np.array([xorA[i, riv_j[i]] for i in range(L)])
    ratio = xor_riv / np.maximum(xor_true, 1.0)
    wrong = disc[off >= 0.62]
    allw = list(wrong) + ss_disc
    # the disc that refutes each of the c strong clones, per letter
    clone_disc = [[round(float(disc[i, j]), 2) for j in topc_idx[i]] for i in range(L)]

    out = dict(
        c_clones_per_letter=C,
        self_iou=[round(float(x), 4) for x in diag],
        rival_iou=[round(float(x), 4) for x in row_riv],
        rival2_iou=[round(float(x), 4) for x in row_2nd],
        rival_c_iou=[round(float(x), 4) for x in riv_c],
        rival_cplus1_iou=[round(float(x), 4) for x in riv_c1],
        rival_idx=[int(x) for x in riv_j],
        topc_idx=[[int(x) for x in v] for v in topc_idx],
        clone_refute_disc_px=clone_disc,
        margin_min=round(float((diag - np.maximum(row_riv, col_riv)).min()), 4),
        unique_bijection=uniq, rowwise_strict_argmax=uniq_row,
        exhaustive_unique=exh,
        exhaustive_margin=round(float(val - sec), 4),
        min_disc_radius_px=round(float(min(allw)) if allw else 99.0, 2),
        min_shadow_shadow_disc_px=round(float(min(ss_disc)) if ss_disc else 99.0, 2),
        min_family_disc_px=round(float(min(fam_disc)) if fam_disc else 99.0, 2),
        disc_p10_px=round(float(np.percentile(allw, 10)) if allw else 99.0, 2),
        true_pair_disc_px=[round(float(x), 2) for x in true_disc],
        xor_ratio=[round(float(x), 2) for x in ratio],
        outline_frac_rival=[round(float(ofrac[i, riv_j[i]]), 3) for i in range(L)],
        n_confusable_pairs=int((off >= 0.62).sum()) + ss_pairs,
        area_rule_hits=int(area_rule),
        n_strong_calls=L * (C + 1),
        n_judgments=L * NSHAD)
    rivlist = []
    for i in range(L):
        js = sorted([j for j in range(NSHAD) if j != i], key=lambda j: -best[i, j])
        rivlist.append([dict(shadow=int(j), iou=round(float(best[i, j]), 4),
                             refute_disc_px=round(float(disc[i, j]), 1),
                             xor_px=int(xorA[i, j])) for j in js[:C + 1]])
    out["rivals_per_letter"] = rivlist
    return out


# --------------------------------------------------------------- LADDER (measured)
# Everything below is re-derived FROM THE RENDERED PNGs by verifier.py, which reads
# nothing from the generator except the per-rung canvas constants and the
# labelling convention (patterns get A,B,C.. and shadows 1,2,3.. in reading
# order); it decides pattern-vs-shadow from colour saturation, re-extracts every
# silhouette from pixels, and re-solves the matching from scratch.
#
#  rung    canvas   tok  cells  tile  LxN     c  L*c  strong calls  injections
#  hard    728x476  442  4x7=28  92   6x22    2   12       18        5.37e7
#  xhard   728x616  572  5x7=35  92   6x29    3   18       24        3.42e8
#  xxhard  728x616  572  5x7=35  92   5x30    4   20       25        1.71e7
#
#  "strong calls" = L*(c+1) = how many times a reader has to hold a coloured
#  pattern against a shadow that is inside the 0.86-0.945 band.  With one clone
#  per letter it is L*2 = 12 / 16 / 20, and when only ONE of each pair carries a
#  blemish the call can be made without looking at the pattern at all.
#
#  PER-JUDGMENT DIFFICULTY, measured, identical on all three rungs by design:
#  rung    clone IoU lo/med/hi   1st non-clone  refute disc  clone-clone  true pair
#  hard    0.861/0.895/0.919     <= 0.812       >= 8.0 px    >= 8.0 px    <= 1.00 px
#  xhard   0.865/0.899/0.918     <= 0.819       >= 8.0 px    >= 8.0 px    <= 1.00 px
#  xxhard  0.864/0.895/0.922     <= 0.818       >= 8.0 px    >= 8.0 px    <= 1.00 px
#
#  The 8:1 separation the whole task rests on is intact and unchanged: the right
#  shadow differs from its pattern by at most a 1.0 px sliver (antialiasing), and
#  EVERY wrong candidate -- including every clone against every other clone in
#  its own family -- is refuted by a solid disc of at least 8.0 px.  Nothing was
#  made smaller to make this rung harder; there is simply more of it to read.
#
# UNIQUENESS IS PROVED THREE WAYS on all 18 images: the strict-row-argmax
# theorem, an exact Hungarian second-best assignment, and a COMPLETE
# branch-and-bound over every injection (7-19 nodes suffice because
# the bound is admissible, and it is unit-tested against exhaustive enumeration).
# On xxhard, an unpruned scoring of all 17,100,720 injections of every image
# agrees; min best-second gap 0.0751.
#
# WHERE THIS AXIS ENDS.  Accuracy is p(c)^L and the pixel budget caps L*c: every
# letter needs c+1 cells for its family plus one for the pattern, and an
# answerless family plus a lone shadow cost 3 more, so
#     cells = L(c+2) + 3 <= 35   ->   L <= 32/(c+2)   ->   L*c <= 32c/(c+2) < 32.
# Back-solving the measured 83% at (L=6, c=1) gives p1 = 0.9694 per strong call, so
# under independence the whole-answer accuracy is p1^(L*c) and the floor of the
# ENTIRE axis is p1^24 ~ 0.47.  c = 6 (L = 4, L*c = 24) was built and measured:
# with a 92 px tile the icon pool cannot carry six well-separated blemishes on
# one contour -- 8000 seeds, 49 icons struck out of the pool, ZERO images in
# 854 s -- so L*c = 20 is where the top rung actually lands.
#
# The cache tables are built by
#     python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_cache
# (see docs/ASSETS.md); items are generated through ../adapter.py (python -m visro).
