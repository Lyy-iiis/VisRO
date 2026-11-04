"""Independent verifier for rotation_patterns.

Imports NOTHING from tasks/planar_transformations/rotation_patterns/generator (the generator) and never calls its
answer code.  Inputs: the record (question text, answer, meta, core_key) and the
delivered image.  Three routes, each enough to fix the gold on its own:

  PIXELS -> STRUCTURE.  Every figure (base + S candidates) is decoded from the
    delivered pixels: the hub colour gives the ink threshold; along each of the
    16 spoke rays (j=0 straight up, clockwise) a perpendicular width profile of
    the coloured ink finds the single bead and its radius; all 16*(S+1) bead
    radii are clustered into the L lengths the question states, and the cluster
    centres must coincide with the L faint grey circles found independently on
    the between-spoke rays.  Each candidate must then equal exactly one cyclic
    rotation of the base word; the question list is answered from that table.
  PIXELS -> ROTATION.  The base figure crop is literally rotated clockwise by each
    question's target angle (PIL, bicubic) and compared with every candidate
    crop: the gold figure must be the unique close match and a 'none' question
    must have no close match.  This pins the clockwise convention independently
    of the ray decoder.
  META -> SOLVER.  A fresh cyclic-word solver over meta (word, shifts, plan).

Plus: the question text is re-rendered from the parsed (S, L, questions) with
this module's own copy of the published wording and must match byte for byte;
the question must print no answer-shaped format example; the tier's declared
form (body, S, number of questions/none/chains, min_gap) is checked; so is the
tier's declared word SOURCE: L by index for easy, meta.pool/pool_md5, the
decoded base figure's membership in that pool (shipped data, md5-pinned here)
and, for xhard, the near-twin property d(8) in {6,8,10} == min Hamming
distance; the index badges and the 'base' label are read back by
template-matching DejaVu Sans Bold; core_key is recomputed from the decoded
base figure.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import functools
import gzip
import hashlib
import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from visro.seeding import digest

CORRECTNESS = ("The gold is recomputed three ways: (1) every figure is decoded from the delivered pixels "
               "(16 rays, one bead per ray, bead radii clustered into the stated L lengths and matched to the L "
               "grey guide circles) and each question is answered from the decoded rotation table; (2) the base "
               "crop is rotated clockwise by each asked angle and matched against every candidate crop; (3) a "
               "fresh cyclic-word solver runs on meta (word, shifts, plan). All three must equal the record's "
               "answer, and the question text must re-render exactly from the parsed parameters.")
UNIQUENESS = ("The decoded base word is primitive (its 16 rotations are pairwise different, Hamming distance "
              ">= 3), every candidate equals exactly one rotation, no two candidates are equal and none is the "
              "0 or 180 deg rotation, so each asked angle has at most one matching figure; a 'none' question's "
              "angle matches no candidate in both the decoded table and the rotated-pixel comparison, and every "
              "bead sits within 0.25 level steps of its length so no misread spoke can create a second match.")

TASK = "rotation_patterns"
K = 16
BOX = 168
M, GAP, COLS = 10, 6, 7
CANVAS = {(1232, 224): dict(rows=1, slots=6), (1232, 364): dict(rows=2, slots=12)}
R_IN = 26.0          # bars start at the hub edge
RIM_BAND = (73.8, 81.0)  # outer rim ring (same colour, concentric), measured inner/outer edge 74.9/79.6 px
FONT = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')
POOLS_DIR = (_ROOT / 'tasks/planar_transformations/rotation_patterns/generator/pools')

# The declared form of each tier (TIER_DEFINITIONS in the adapter).
TIER_FORM = {
    "easy":   dict(body="plain", S=6, n=2, n_none=0, n_chain=0, L=(3, 4), min_gap=2),
    "medium": dict(body="plain", S=6, n=3, n_none=0, n_chain=0, L=(5,), min_gap=1),
    "hard":   dict(body="plain", S=12, n=9, n_none=0, n_chain=0, L=(5,), min_gap=1),
    "xhard":  dict(body="enum", S=12, n=12, n_none=2, n_chain=0, L=(5,), min_gap=1),
    "xxhard": dict(body="enum", S=12, n=12, n_none=2, n_chain=3, L=(5,), min_gap=1),
}

MAX_SNAP = 0.25        # bead radius distance from its length cluster, in level steps
MIN_HAM = 3            # min Hamming distance between distinct rotations of the base word
# Rotated-pixel route: the base crop is rotated (bicubic) by the asked angle and compared with a
# candidate crop inside the figure disc; the score is the largest 8-connected blob of pixels whose
# max |RGB diff| exceeds 60.  Calibrated on 75 released medium/hard/xhard items, every legal angle
# against every candidate: 750 true pairs score 0..2 px, 9,750 wrong pairs 113..248 px.  A score
# <= ROT_CLOSE is a match, >= ROT_FAR is a different figure, anything between is reported unclear.
# (A mean-|diff| score is too weak for xhard's near-twins.)
ROT_CLOSE = 30
ROT_FAR = 60

# The declared word SOURCE of each tier (TIER_DEFINITIONS in the adapter), kept here as data: the
# pool each item must come from and the md5 of that pool's canonical json.  The pools are read as
# DATA from the shipped tasks/planar_transformations/rotation_patterns/generator/pools/*.json.gz (md5 asserted against this table);
# no generator code is imported.  easy takes L=3 on every index with index % 16 == 0, else L=4.
POOL_MD5 = {
    "_pool_L3w8full.json": "ff47aaddb350b4d274bd1a6a6abaff9b",
    "_pool_L4w8full.json": "2b69aa46d939b00416a2888936442525",
    "_pool_L5w8.json": "ad8748ba4663adc159594366eaf8c016",
    "_pool_L5xwd.json": "6a81313456eea76c93f0b23ff339ddfb",
    "_pool_L5m.json": "94560be3496e09a86fef44e904699672",
}
EASY_PERIOD = 16
XHARD_D8 = (6, 8, 10)       # xhard's near-twin family: d(8) in {6,8,10} and the 180 deg twin is the closest


def tier_source(tier: str, index: int):
    """(L, pool) a record of `tier` at candidate `index` must have been drawn from."""
    if tier == "easy":
        return (3, "_pool_L3w8full.json") if index % EASY_PERIOD == 0 else (4, "_pool_L4w8full.json")
    return {"medium": (5, "_pool_L5m.json"), "hard": (5, "_pool_L5w8.json"),
            "xhard": (5, "_pool_L5xwd.json"), "xxhard": (5, "_pool_L5w8.json")}[tier]


@functools.lru_cache(maxsize=None)
def pool_words(name: str):
    """The shipped pool as a frozenset of words; md5 of its canonical json must equal POOL_MD5."""
    raw = gzip.decompress((POOLS_DIR / (name + ".gz")).read_bytes())
    if hashlib.md5(raw).hexdigest() != POOL_MD5[name]:
        raise ValueError(f"shipped pool {name}: md5 {hashlib.md5(raw).hexdigest()} != {POOL_MD5[name]}")
    d = json.loads(raw)
    return int(d["L"]), frozenset(tuple(int(c) for c in w) for w in d["words"])


def variants(w, L):
    """Every rotation of the word, its mirror, its level inversion and the mirror of that."""
    rev = tuple(w[(-i) % K] for i in range(K))
    seeds = {tuple(w), rev, tuple(L - 1 - c for c in w), tuple(L - 1 - c for c in rev)}
    return {tuple(sd[(j - s) % K] for j in range(K)) for sd in seeds for s in range(K)}


# ------------------------------------------------------------------ wording
# The published wording of rotation_patterns (compact form: every convention once; at most 240 tokens
# under the vicuna (+BOS) and PaliGemma tokenizers for every plan any tier can draw).  This is the verifier's
# own copy; the record's question must equal render_question(parse_question(...)) byte for byte.
#   plain (easy, medium, hard): "Which figure is the base turned clockwise by each angle: A, B, C? Answer the
#       figure numbers in order, comma-separated."
#   enum (xhard, xxhard): "Which figure is the base[ (or the named figure)] turned clockwise by each angle below?
#       Write none if no figure is. Answer in order, comma-separated." followed by one line per question, "A" or
#       "figure k+A" (figure k turned a further A).
def _fmt(deg_steps: int) -> str:
    return f"{360.0 * deg_steps / K:g}"


def _stem(S, L, body):
    return (f"Figures 1-{S} are the bold-framed base (far left) turned clockwise by different multiples of 22.5 "
            f"degrees; the bead lengths on the {K} spokes are the code. ")


def render_question(S, L, body, qs):
    """qs: list of ("abs", None, step) / ("chain", ref, step).  The published wording."""
    items = [_fmt(st) if kind == "abs" else f"figure {ref}+{_fmt(st)}" for kind, ref, st in qs]
    if body == "plain":
        return (_stem(S, L, body) + f"Which figure is the base turned clockwise by each angle: {', '.join(items)}? "
                "Answer the figure numbers in order, comma-separated.")
    who = "the base (or the named figure)" if any(kind == "chain" for kind, _, _ in qs) else "the base"
    return (_stem(S, L, body) + f"Which figure is {who} turned clockwise by each angle below? Write none if no "
            "figure is. Answer in order, comma-separated.\n" + "\n".join(items))


_ANGLE = r"\d+(?:\.\d+)?"


def _deg_to_step(txt: str) -> int:
    a = float(txt)
    st = a / 22.5
    if abs(st - round(st)) > 1e-9 or not (0 < round(st) < K):
        raise ValueError(f"angle {txt} is not a multiple of 22.5 in (0, 360)")
    return int(round(st))


def parse_question(q: str, L: int):
    """L (the number of bead lengths) is not printed; it comes from the record's meta."""
    m = re.match(r"Figures 1-(\d+) are the bold-framed base ", q)
    if not m:
        raise ValueError("stem: no 'Figures 1-S are the bold-framed base'")
    S = int(m.group(1))
    qs = []
    m = re.search(r"Which figure is the base turned clockwise by each angle: ([^?\n]*)\? ", q)
    if m:
        body = "plain"
        for a in m.group(1).split(", "):
            if not re.fullmatch(_ANGLE, a):
                raise ValueError(f"plain body: bad angle token {a!r}")
            qs.append(("abs", None, _deg_to_step(a)))
    else:
        body = "enum"
        head, sep, rest = q.partition("\n")
        if not sep:
            raise ValueError("enumerated body: no question lines")
        for line in rest.split("\n"):
            m1 = re.fullmatch(_ANGLE, line)
            m2 = re.fullmatch(rf"figure (\d+)\+({_ANGLE})", line)
            if m1:
                qs.append(("abs", None, _deg_to_step(line)))
            elif m2:
                qs.append(("chain", int(m2.group(1)), _deg_to_step(m2.group(2))))
            else:
                raise ValueError("unrecognised question line: " + line[:80])
    if not qs:
        raise ValueError("no questions parsed")
    if render_question(S, L, body, qs) != q:
        raise ValueError("question text does not re-render from its parsed parameters (form mismatch)")
    return dict(S=S, L=L, body=body, qs=qs)


# ------------------------------------------------------------------- layout
def tile_origins(w, h):
    if (w, h) not in CANVAS:
        raise ValueError(f"canvas {w}x{h} is not a documented layout")
    rows, slots = CANVAS[(w, h)]["rows"], CANVAS[(w, h)]["slots"]
    box = int(min((h - 2 * M - (rows - 1) * GAP) / rows, (w - 2 * M - (COLS - 1) * GAP) / COLS))
    assert box == BOX, box
    y0 = M + (h - 2 * M - (rows * box + (rows - 1) * GAP)) // 2
    base = (M, y0)
    cands = []
    for i in range(slots):
        c, r = i % (COLS - 1) + 1, i // (COLS - 1)
        cands.append((M + c * (box + GAP), y0 + r * (box + GAP)))
    return base, cands


# ------------------------------------------------------------------- decode
_RS = np.arange(28.5, 73.01, 0.25)    # hub edge .. just inside the rim
_TS = np.arange(-7.0, 7.01, 0.25)


def _ray_angles(offset=0.0):
    return np.array([-math.pi / 2 + 2 * math.pi * (j + offset) / K for j in range(K)])


def decode_tile(tile: np.ndarray):
    """tile: (168,168,3) uint8.  Returns dict(colour, radii[16], problems)."""
    a = tile.astype(np.float64)
    sat = a.max(axis=2) - a.min(axis=2)
    val = a.mean(axis=2)
    n = BOX
    ys, xs = np.mgrid[0:n, 0:n]
    c0 = (n - 1) / 2.0
    rr = np.hypot(xs - c0, ys - c0)
    probs = []
    hub = rr < 18
    colour = np.median(a[hub], axis=0)
    s_full = float(colour.max() - colour.min())
    if s_full < 50:
        return dict(colour=colour, radii=None, problems=[f"hub is not a saturated colour ({colour.tolist()})"])
    thr = 0.5 * s_full
    ink = sat > thr
    # centre from the rim ring (concentric, same colour)
    rim = ink & (rr > RIM_BAND[0]) & (rr < RIM_BAND[1] + 2)
    if rim.sum() < 200:
        return dict(colour=colour, radii=None, problems=["no rim ring"])
    cy, cx = ys[rim].mean(), xs[rim].mean()
    if abs(cx - c0) > 0.6 or abs(cy - c0) > 0.6:
        probs.append(f"figure centre off by ({cx - c0:.2f},{cy - c0:.2f}) px")
    ang = _ray_angles()
    ux, uy = np.cos(ang), np.sin(ang)
    # sample points [ray, r, t]
    X = cx + _RS[None, :, None] * ux[:, None, None] - _TS[None, None, :] * uy[:, None, None]
    Y = cy + _RS[None, :, None] * uy[:, None, None] + _TS[None, None, :] * ux[:, None, None]
    S_ = ndimage.map_coordinates(sat, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
    width = (S_ > thr).sum(axis=2) * 0.25                       # [ray, r]
    centre_line = S_[:, :, len(_TS) // 2] > thr                 # t = 0
    radii = []
    for j in range(K):
        wj = width[j]
        on = wj > 6.5
        lab, nrun = ndimage.label(on)
        if nrun != 1:
            probs.append(f"ray {j}: {nrun} bead-width runs (expected exactly 1)")
            radii.append(float("nan"))
            continue
        idx = np.flatnonzero(lab == 1)
        p = 0.5 * (_RS[idx[0]] + _RS[idx[-1]])
        radii.append(float(p))
        # solid bar from the hub to the bead, and nothing between bead and rim
        bar = centre_line[j][_RS <= p]
        if not bar.all():
            probs.append(f"ray {j}: bar not solid between hub and bead")
        beyond = wj[_RS > p + 7.0]
        if beyond.size and beyond.max() > 1.0:
            probs.append(f"ray {j}: ink beyond the bead")
    # every coloured pixel must belong to hub, rim or a spoke corridor
    th = np.arctan2(ys - cy, xs - cx)
    r2 = np.hypot(xs - cx, ys - cy)
    jn = np.rint((th + math.pi / 2) / (2 * math.pi / K)).astype(int) % K
    dth = th - (-math.pi / 2 + 2 * math.pi * jn / K)
    dth = (dth + math.pi) % (2 * math.pi) - math.pi
    perp = np.abs(r2 * np.sin(dth))
    reach = np.array([r if r == r else 0.0 for r in radii])[jn] + 7.0
    allowed = (r2 < R_IN + 2.0) | ((r2 > RIM_BAND[0]) & (r2 < RIM_BAND[1])) | ((perp < 7.0) & (r2 < reach))
    stray = ink & ~allowed
    if stray.sum() > 3:
        probs.append(f"{int(stray.sum())} coloured pixels outside the hub/rim/spokes")
    return dict(colour=colour, radii=radii, problems=probs, centre=(cx, cy), val=val, sat=sat, s_full=s_full)


def grey_rings(dec) -> list[list[float]]:
    """Ring radii found on each between-spoke ray (grey, unsaturated, darker than white)."""
    val, sat = dec["val"], dec["sat"]
    cx, cy = dec["centre"]
    out = []
    rs = np.arange(28.5, 73.01, 0.25)            # hub edge .. rim edge, exclusive
    for a in _ray_angles(0.5):
        X, Y = cx + rs * math.cos(a), cy + rs * math.sin(a)
        v = ndimage.map_coordinates(val, [Y, X], order=1)
        s = ndimage.map_coordinates(sat, [Y, X], order=1)
        g = (v < 238) & (s < 0.3 * dec["s_full"])
        lab, nrun = ndimage.label(g)
        out.append([float(rs[lab == k].mean()) for k in range(1, nrun + 1)])
    return out


def cluster_levels(radii: list[float], L: int):
    """Split sorted radii at the L-1 largest gaps.  Returns (centres, labels, worst snap in steps, problem)."""
    r = np.asarray(radii, dtype=float)
    order = np.argsort(r)
    rs = r[order]
    gaps = np.diff(rs)
    if L < 2 or len(rs) < L:
        return None, None, None, f"cannot cluster {len(rs)} radii into {L}"
    cut = np.sort(np.argsort(gaps)[-(L - 1):])
    groups = np.split(np.arange(len(rs)), cut + 1)
    centres = np.array([rs[g].mean() for g in groups])
    labels_sorted = np.concatenate([[k] * len(g) for k, g in enumerate(groups)])
    labels = np.empty(len(r), dtype=int)
    labels[order] = labels_sorted
    step = float(np.min(np.diff(centres)))
    snap = float(np.max(np.abs(r - centres[labels])) / step)
    prob = None
    if snap > MAX_SNAP:
        prob = f"bead radius {snap:.2f} level steps from its length (> {MAX_SNAP}); lengths not cleanly {L}"
    return centres, labels, snap, prob


# ------------------------------------------------------------------- badges
@functools.lru_cache(maxsize=64)
def _label_mask(text: str):
    font = ImageFont.truetype(str(FONT), 13)
    im = Image.new("L", (60, 40), 255)
    d = ImageDraw.Draw(im)
    d.text((10, 10), text, font=font, fill=20)
    a = np.asarray(im) < 90
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@functools.lru_cache(maxsize=4)
def _base_label_mask():
    font = ImageFont.truetype(str(FONT), 15)
    im = Image.new("L", (90, 40), 255)
    ImageDraw.Draw(im).text((10, 10), "base", font=font, fill=35)
    a = np.asarray(im) < 90
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


LABEL_AT = (4, BOX + 13)    # 'base' is drawn at (bx+2, by+BOX+10); DejaVu's bearing/ascent put its ink here


def read_base_label(canvas: np.ndarray, bx, by):
    """Best IoU of the word 'base' (DejaVu Sans Bold 15) with the dark ink under the base figure, over placements
    within 2 px of where it is drawn, and that placement.  The window is the template plus a 3 px margin, so stray
    ink next to the word lowers the score; a fixed window (not a bounding box) keeps one flipped edge pixel from
    changing the comparison's shape (the audit's sigma-4 noise control)."""
    t = _base_label_mask()
    th, tw = t.shape
    ex, ey = bx + LABEL_AT[0], by + LABEL_AT[1]
    pad = 3
    dark = canvas.astype(np.float64).mean(axis=2) < 90
    dark = np.pad(dark, pad + 2, constant_values=False)
    o = pad + 2
    win = dark[o + ey - pad:o + ey + th + pad, o + ex - pad:o + ex + tw + pad]
    best, where = 0.0, None
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            T = np.zeros_like(win)
            T[pad + dy:pad + dy + th, pad + dx:pad + dx + tw] = t
            iou = float((T & win).sum() / max((T | win).sum(), 1))
            if iou > best:
                best, where = iou, (ex + dx, ey + dy)
    return best, where


def read_badge(canvas: np.ndarray, x, y, S):
    # inside the badge box, clear of its outline and >= 84 px from the dial centre (rim ends at 79.6)
    reg = canvas[y + 2:y + 20, x + 2:x + 28].astype(np.float64).mean(axis=2) < 90
    ys, xs = np.nonzero(reg)
    if not len(ys):
        return None, 0.0
    got = reg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    best, score = None, 0.0
    for k in range(1, S + 1):
        t = _label_mask(str(k))
        if t.shape != got.shape:
            continue
        iou = float((t & got).sum() / max((t | got).sum(), 1))
        if iou > score:
            best, score = k, iou
    return best, score


# ------------------------------------------------------------------ helpers
def _rot(w, s):
    return tuple(w[(j - s) % K] for j in range(K))


def _canon(w):
    return min(tuple(w[(i + s) % K] for i in range(K)) for s in range(K))


def _disc_mask():
    ys, xs = np.mgrid[0:BOX, 0:BOX]
    c = (BOX - 1) / 2.0
    return np.hypot(xs - c, ys - c) < 82.5


_DISC = _disc_mask()


def _rot_diff(base_tile: Image.Image, cand: np.ndarray, steps: int) -> int:
    """Largest blob (px) of strong change between the base rotated clockwise by steps*22.5 deg and cand."""
    rb = np.asarray(base_tile.rotate(-22.5 * steps, resample=Image.BICUBIC)).astype(np.float64)
    m = (np.abs(rb - cand).max(axis=2) > 60) & _DISC
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    return int(ndimage.sum(m, lab, range(1, n + 1)).max()) if n else 0


def answer_from_table(qs, shifts):
    out = []
    for kind, ref, st in qs:
        if kind == "abs":
            t = st
        else:
            if not (1 <= ref <= len(shifts)):
                return None, f"chain root figure {ref} does not exist"
            t = (shifts[ref - 1] + st) % K
        hits = [i + 1 for i, s in enumerate(shifts) if s == t]
        if len(hits) > 1:
            return None, f"angle {_fmt(t)} matches {len(hits)} figures"
        out.append(str(hits[0]) if hits else "none")
    return out, None


# ------------------------------------------------------------------- verify
def verify(record: dict, image) -> list[str]:
    P = []
    meta = record.get("meta") or {}
    tier = record.get("tier")
    if record.get("task") != TASK:
        P.append(f"task {record.get('task')!r}")
    if tier not in TIER_FORM:
        return P + [f"unknown tier {tier!r}"]
    if record.get("answer_type") != "token-list":
        P.append("answer_type must be token-list")
    form = TIER_FORM[tier]
    try:
        pq = parse_question(record["question"], int(record["meta"]["L"]))
    except (ValueError, KeyError) as exc:
        return P + [f"question: {exc}"]
    S, L, qs, body = pq["S"], pq["L"], pq["qs"], pq["body"]

    # --- declared form of the tier
    n_chain = sum(k == "chain" for k, _, _ in qs)
    if body != form["body"] or S != form["S"] or len(qs) != form["n"] or n_chain != form["n_chain"] \
            or L not in form["L"]:
        P.append(f"form {body}/S={S}/n={len(qs)}/chains={n_chain}/L={L} is not the declared {tier} form")
    targets_abs = [st for k, _, st in qs if k == "abs"]
    if any(st in (0, K // 2) for st in targets_abs):
        P.append("question asks for 0 or 180 deg, which the stem excludes")

    # --- answer string format
    ans = record.get("answer", "")
    toks = ans.split(",")
    if len(toks) != len(qs) or any(t != t.strip() or not t for t in toks):
        P.append(f"answer has {len(toks)} tokens for {len(qs)} questions")
    ok_tok = {str(i) for i in range(1, S + 1)} | ({"none"} if body == "enum" else set())
    if any(t not in ok_tok for t in toks):
        P.append(f"answer token outside {'1..S|none' if body == 'enum' else '1..S'}")
    # the question carries no format example: no answer-shaped (comma-joined, no space) token list
    if re.search(r"\w,\w", record["question"]):
        P.append("question contains an answer-shaped token list (format example)")

    # --- declared word source of the tier (L by index, pool, pool md5)
    try:
        index = int(record["source"]["index"])
        want_L, want_pool = tier_source(tier, index)
    except (KeyError, TypeError, ValueError) as exc:
        return P + [f"record source/index unreadable: {exc!r}"]
    if L != want_L:
        P.append(f"{tier} index {index} must use L={want_L} (declared source), question says L={L}")
    if meta.get("pool") != want_pool or meta.get("pool_md5") != POOL_MD5[want_pool]:
        P.append(f"meta pool {meta.get('pool')!r}/{meta.get('pool_md5')!r} is not the declared source "
                 f"{want_pool}/{POOL_MD5[want_pool]}")

    # --- image layout
    im = image.convert("RGB")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    try:
        base_o, cand_o = tile_origins(w, h)
    except (ValueError, AssertionError) as exc:
        return P + [f"image: {exc}"]
    if len(cand_o) != S:
        P.append(f"stem says {S} numbered figures, canvas holds {len(cand_o)}")
        return P
    bx, by = base_o
    frame = arr[by - 4:by, bx:bx + BOX].astype(np.float64)
    if abs(frame.mean() - 45) > 15:
        P.append("base figure has no bold frame")
    lab_iou, _lab_at = read_base_label(arr, bx, by)
    if lab_iou < 0.9:
        P.append(f"no 'base' label under the base figure (best template iou {lab_iou:.2f} within 2 px of "
                 f"{(bx + LABEL_AT[0], by + LABEL_AT[1])})")
    for i, (x, y) in enumerate(cand_o):
        k, score = read_badge(arr, x, y, S)
        if k != i + 1 or score < 0.9:
            P.append(f"badge of slot {i + 1} reads {k} (iou {score:.2f})")
            break

    # --- decode every figure
    crops = [arr[by:by + BOX, bx:bx + BOX]] + [arr[y:y + BOX, x:x + BOX] for x, y in cand_o]
    decs = [decode_tile(c) for c in crops]
    for i, d in enumerate(decs):
        for p in d["problems"]:
            P.append(f"{'base' if i == 0 else f'figure {i}'}: {p}")
    if any(d["radii"] is None or any(r != r for r in d["radii"]) for d in decs):
        return P
    cols = np.array([d["colour"] for d in decs])
    if np.abs(cols - cols[0]).max() > 12:
        P.append("figures are not all drawn in one colour")
    allr = [r for d in decs for r in d["radii"]]
    centres, labels, snap, prob = cluster_levels(allr, L)
    if prob:
        P.append(prob)
    if centres is None:
        return P
    words = [tuple(int(x) for x in labels[k * K:(k + 1) * K]) for k in range(S + 1)]
    base = words[0]
    if len(set(base)) != L:
        P.append(f"base figure uses {len(set(base))} lengths, stem says {L}")
    # the L faint grey circles
    ring_counts, ring_r = [], []
    for d in decs[:3]:
        for rays in grey_rings(d):
            ring_counts.append(len(rays))
            if len(rays) == L:
                ring_r.append(rays)
    mode = max(set(ring_counts), key=ring_counts.count)
    if mode != L or len(ring_r) < 0.75 * len(ring_counts):
        P.append(f"grey guide circles: {mode} found per ray (stem says {L})")
    else:
        rr = np.median(np.array(ring_r), axis=0)
        if np.abs(rr - centres).max() > 1.5:
            P.append(f"bead lengths {np.round(centres, 1).tolist()} do not sit on the grey circles "
                     f"{np.round(rr, 1).tolist()}")

    # --- uniqueness of the rotation table
    rots = [_rot(base, s) for s in range(K)]
    hams = [sum(a != b for a, b in zip(rots[0], rots[s])) for s in range(K)]
    mh = min(hams[1:])
    if len(set(rots)) != K:
        P.append("base figure has rotational symmetry: two angles give the same figure")
    elif mh < MIN_HAM:
        P.append(f"two rotations of the base differ in only {mh} spokes")
    # --- the tier's word source: the decoded base figure must come from the declared pool
    try:
        pool_L, pool_set = pool_words(want_pool)
    except (OSError, ValueError, KeyError) as exc:
        P.append(f"cannot read the declared pool {want_pool}: {exc}")
    else:
        if pool_L != L or not (variants(base, L) & pool_set):
            P.append(f"decoded base figure is not a figure of the declared pool {want_pool} "
                     "(up to rotation, mirror and level inversion)")
    if tier == "xhard" and not (hams[K // 2] in XHARD_D8 and mh == hams[K // 2]):
        P.append(f"xhard base is not a near-twin figure: d(8)={hams[K // 2]}, min Hamming {mh} "
                 f"(declared d(8) in {XHARD_D8} and the 180 deg twin the closest rotation)")
    shifts = []
    for i, cw in enumerate(words[1:], 1):
        m = [s for s in range(K) if rots[s] == cw]
        if len(m) != 1:
            P.append(f"figure {i} equals {len(m)} rotations of the base")
            shifts.append(None)
        else:
            shifts.append(m[0])
    if None in shifts:
        return P
    if len(set(shifts)) != S:
        P.append("two numbered figures show the same rotation")
    if any(s in (0, K // 2) for s in shifts):
        P.append("a numbered figure is the 0 or 180 deg rotation, which the stem excludes")
    pts = [0] + shifts
    gap = min(min((a - b) % K, (b - a) % K) for i, a in enumerate(pts) for b in pts[i + 1:])
    if gap < form["min_gap"]:
        P.append(f"bank angular gap {gap} notches < declared min_gap {form['min_gap']}")

    # --- answer from the decoded table
    got, err = answer_from_table(qs, shifts)
    if err:
        P.append(err)
        return P
    if ",".join(got) != ans:
        P.append(f"pixel-decoded answer {','.join(got)} != gold {ans}")
    if body == "plain" and "none" in got:
        P.append("plain body asks for an angle no figure shows")
    if body == "enum":
        if sum(t == "none" for t in got) != form["n_none"]:
            P.append(f"{sum(t == 'none' for t in got)} none answers, tier declares {form['n_none']}")
        tg = []
        for kind, ref, st in qs:
            tg.append(st if kind == "abs" else (shifts[ref - 1] + st) % K)
        if len(set(tg)) != len(tg):
            P.append("two questions describe the same figure")
        if any(t in (0, K // 2) for t in tg):
            P.append("a question's target is the 0 or 180 deg rotation (a free none)")

    # --- pixels -> rotation route
    base_img = Image.fromarray(np.ascontiguousarray(crops[0]))
    cand = [c.astype(np.float64) for c in crops[1:]]
    tg = [st if kind == "abs" else (shifts[ref - 1] + st) % K for kind, ref, st in qs]
    for qi, (t, tok) in enumerate(zip(tg, got), 1):
        diffs = np.array([_rot_diff(base_img, c, t) for c in cand])
        close = [int(i) + 1 for i in np.flatnonzero(diffs <= ROT_CLOSE)]
        unclear = [int(i) + 1 for i in np.flatnonzero((diffs > ROT_CLOSE) & (diffs < ROT_FAR))]
        want = [] if tok == "none" else [int(tok)]
        if close != want or unclear:
            P.append(f"q{qi}: base rotated by {_fmt(t)} deg matches figures {close}, unclear {unclear} "
                     f"(blob px {sorted(diffs.tolist())[:3]}), gold says {tok}")

    # --- meta route (fresh solver) and cross-checks
    try:
        mw = tuple(int(x) for x in meta["word"])
        msh = [int(s) for s in meta["shifts"]]
        plan = meta["plan"]
        if mw != base:
            P.append("meta word != decoded base word")
        if msh != shifts:
            P.append("meta shifts != decoded shifts")
        if int(meta["L"]) != L or int(meta["slots"]) != S:
            P.append("meta L/slots disagree with the question")
        if np.abs(np.array(meta["colour"], dtype=float) - cols[0]).max() > 12:
            P.append("meta colour != drawn colour")
        mq = []
        for sp in plan:
            if sp["kind"] == "abs":
                mq.append(("abs", None, int(sp["step"]) % K))
            elif sp["kind"] == "chain" and len(sp["steps"]) == 1:
                mq.append(("chain", int(sp["ref"]), int(sp["steps"][0]) % K))
            else:
                mq.append(("unsupported", None, None))
        if mq != qs:
            P.append("meta plan != questions in the text")
        mwords = [_rot(mw, s) for s in msh]
        sol = []
        for sp in plan:
            if sp["kind"] == "abs":
                t = int(sp["step"]) % K
            else:
                t = (msh[int(sp["ref"]) - 1] + sum(int(x) for x in sp["steps"])) % K
            tw = _rot(mw, t)
            hit = [i + 1 for i, cw in enumerate(mwords) if cw == tw]
            sol.append(str(hit[0]) if len(hit) == 1 else ("none" if not hit else "AMBIG"))
            if bool(sp.get("absent")) != (not hit):
                P.append("meta plan 'absent' flag disagrees with the bank")
        if ",".join(sol) != ans:
            P.append(f"meta solver answer {','.join(sol)} != gold {ans}")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        P.append(f"meta incomplete: {exc!r}")

    # --- dedup key is the decoded base figure
    want = digest({"task": TASK, "kind": "base-figure-necklace", "L": L, "necklace": list(_canon(base))})
    if record.get("core_key") != want:
        P.append("core_key is not the necklace of the decoded base figure")
    return P
