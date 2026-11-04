"""count_points_on_lines -- independent verifier.

Imports nothing from tasks/discrimination_grouping/count_points_on_lines/generator or generator/base: every number is
recomputed here from the record (meta.segments_grid, grid_n, canvas) and from
the decoded PNG.

Layers (all must pass):
  Q  question     byte-equal to the declared question, ASCII, and it states the
                  incidence rule, the de-duplication rule and the answer format.
  A  answer       exact integer recompute: a lattice point P is on segment AB iff
                  cross(B-A, P-A) == 0 and 0 <= dot(P-A, B-A) <= |B-A|^2.
                  Per-segment counts sorted descending, ';', size of the union.
                  Must equal the gold string and meta's per_line / on_dots_grid.
                  The question's format example (parsed from the question) may
                  not equal the gold, have as many counts, or share ANY number
                  with it in any position.
  T  tier shape   N, L, canvas, dot radius, line width, near-miss margin and
                  count, the near-miss margin (no OFF dot inside it), segment rules
                  (primitive direction, >=4 dots, length) as the tier table declares,
                  and the anti-shortcut rules: no slope 0/inf/+-1 (G11), per-segment
                  counts not all equal (G12).
  U  uniqueness   (geometry, from meta, in pixels)
                  - every non-incident (dot, segment) pair: capsule distance to
                    the OVERSHOT segment (beyond-tip regions included) minus dot
                    radius minus half line width >= 3 px  [OFF and ON dots];
                  - segments on pairwise distinct infinite lines;
                  - overshot endpoints of different segments >= 0.8*spacing apart (G7);
                  - segments whose directions differ by < 15 deg >= 1.0*spacing apart (G8).
  P  pixels       decode the image without meta: dark components -> N x N
                  lattice; blue ink -> links between lattice-consecutive dots
                  whose straight path is ink outside dot discs; maximal chains
                  with visible overshoot past both end dots are strokes.  The
                  decoded strokes must equal meta's segments (so the decoded
                  answer equals the gold).  Then: every ink pixel lies within
                  1.75 px of the edge of a declared stroke placed on the decoded
                  dot centroids; each stroke's ink axis (least-squares fit)
                  passes within 0.6 px of every ON dot's lattice position; every
                  non-incident dot within 1.2 spacings of a stroke is separated
                  from that stroke's ink by >= 1 px (OFF dots: >= 1 px pure white).
  R  rendering    the PNG equals this module's own rendering of meta (segments
                  overshot 0.35*spacing, dots on top, 3x supersample + LANCZOS):
                  pixel-exact (max channel difference <= 2) under the Pillow
                  versions EXACT_PILLOW lists; under any other Pillow a drift
                  tolerance applies.  In both modes no pixel away from the
                  declared dots and strokes may differ from white by > 16
                  (stray-mark check, independent of the Pillow version).
  K  core_key     equals sha256 of (task, N, segment set reduced modulo the 8
                  symmetries of the lattice and translation), recomputed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

TASK = "count_points_on_lines"

CORRECTNESS = (
    "Recomputes every per-segment dot count and the de-duplicated total by an exact integer cross-product "
    "scan of the N x N lattice against meta.segments_grid, and independently decodes the PNG (dark blobs -> "
    "lattice, blue ink -> maximal chains of ink-linked lattice dots with visible overshoot); the decoded "
    "strokes, the recomputed 'c1,...,cL;total' string, meta and the gold answer must all agree."
)
UNIQUENESS = (
    "All ink lies on the decoded strokes, strokes sit on distinct lattice lines, strokes within 15 deg stay "
    ">=1 spacing apart and stroke ends >=0.8 spacing apart, so the picture has one stroke decomposition; every "
    "dot either lies exactly on a stroke axis (fitted ink axis within 0.6 px of its lattice position) or clears the "
    "overshot stroke (capsule distance, tips included) by >=3 px with a measured white gap, so each incidence "
    "has one reading and the canonical sorted string is the only valid answer."
)

QUESTION = (
    'Blue segments are drawn over a grid of black dots; a segment passes through a dot only if it'
    " crosses its centre. Answer each segment's dot count in descending order, comma-separated, "
    "then ';' and the number of distinct dots covered, no spaces, e.g. 2,2;3."
)
# conventions the question must state (checked separately from byte equality so
# that a future rewording cannot silently drop one)
REQUIRED_PHRASES = (
    "grid of black dots", "Blue segments", "only if it crosses its centre", "descending order",
    "comma-separated", "then ';'", "distinct dots covered", "no spaces",
)
EXAMPLE_ANSWER = "2,2;3"
EXAMPLE_RE = re.compile(r"e\.g\. (\S+?)\.\s*\Z")

# tier table (restated here; see TIER_DEFINITIONS in the adapter)
TIERS = {
    "easy":   dict(N=11, L=3,  size=448, dot_r_frac=0.16, band=(0.40, 0.44), nearmiss=2),
    "medium": dict(N=13, L=4,  size=448, dot_r_frac=0.12, band=(0.28, 0.32), nearmiss=4),
    "hard":   dict(N=19, L=12, size=672, dot_r_frac=0.10, band=(0.24, 0.26), nearmiss=9),
    "xhard":  dict(N=20, L=11, size=672, dot_r_frac=0.10, band=(0.24, 0.26), nearmiss=10),
    "xxhard": dict(N=20, L=13, size=672, dot_r_frac=0.10, band=(0.24, 0.26), nearmiss=10),
}
MARGIN_CANVAS = 0.08
LW_FRAC, MIN_LW = 0.05, 2.0
MIN_DOT_DIA = 6.0
OVERSHOOT = 0.35
MIN_CLEAR_PX = 3.0
EP_SEP_FRAC = 0.80
PARA_ANG_DEG, PARA_SEP_FRAC = 15.0, 1.0
NEARMISS_BAND = 1.7
MAX_PRIM = 5
MIN_STEPS = 3                  # >= 4 dots per segment
MIN_LEN_FRAC = 0.55            # k*|(p,q)| >= 0.55*(N-1)

# pixel-layer thresholds (calibrated on the release sample, see docs)
PIX_POS_TOL = 0.75             # decoded dot centroid vs declared lattice (after the render offset)
RENDER_OFFSET = -0.5           # PIL draws at continuous coords; pixel-index centroids sit 0.5px lower
INK_FRAC = 0.25                # pixel counts as ink if (B-R) >= 25% of the line colour's (B-R)
EXTRA_INK_TOL = 1.75           # max distance (px) of an ink pixel from a stroke edge
AXIS_TOL = 0.6                 # fitted ink axis vs ON-dot centroid (px)
MIN_GAP_PX = 1.0               # dot edge -> this stroke's ink, non-dot samples
MIN_WHITE_PX = 1.0             # pure-white run between an OFF dot and a stroke (gate G4's floor)
WHITE_MIN_CH = 200             # "white" = every channel >= this
# R layer.  The re-render is pixel-identical (max diff 0) to the PNG under the Pillow
# versions below (12.3.0 pinned, 12.2.0 checked), so there the PNG must match exactly
# (2 levels of slack for platform rounding).  Under any other Pillow the drift tolerance
# applies instead.
EXACT_PILLOW = ("12.2.0", "12.3.0")
RERENDER_EXACT_MAX = 2         # pinned Pillow: max channel difference
RERENDER_MAX = 64              # other Pillow: max channel difference ...
RERENDER_FRAC = 0.002          # ... and at most 0.2% of pixels differing by > 16
STRAY_DEV = 16                 # any pixel with 255 - min(R,G,B) > this must lie within
STRAY_DOT_PAD = 1.5            # dot_r + 1.5 px of a lattice dot or
STRAY_STROKE_PAD = 1.5         # lw/2 + 1.5 px of a declared overshot stroke (max LANCZOS ringing outside: 11)


# ----------------------------------------------------------------------------- helpers
def _digest(value) -> str:
    s = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()


def canonical_form(segs, n):
    """Segment set modulo the 8 lattice symmetries and translation (smallest image after
    moving the endpoints' bounding box to the origin)."""
    m = n - 1
    ops = (lambda x, y: (x, y), lambda x, y: (m - x, y), lambda x, y: (x, m - y),
           lambda x, y: (m - x, m - y), lambda x, y: (y, x), lambda x, y: (m - y, x),
           lambda x, y: (y, m - x), lambda x, y: (m - y, m - x))
    best = None
    for op in ops:
        pts = [(op(*a), op(*b)) for a, b in segs]
        ox = min(v[0] for s in pts for v in s)
        oy = min(v[1] for s in pts for v in s)
        img = sorted(tuple(sorted(((a[0] - ox, a[1] - oy), (b[0] - ox, b[1] - oy)))) for a, b in pts)
        if best is None or img < best:
            best = img
    return [[list(a), list(b)] for a, b in best]


def expected_core_key(segs, n) -> str:
    return _digest([TASK, int(n), canonical_form(segs, n)])


def example_overlap(answer, example):
    """Ways a format example gives away part of `answer`: equal, as many counts, or any number
    in common (any position -- covers a shared first count, prefix, pair, total)."""
    if answer == example:
        return ["equals the gold"]
    nums = []
    for s in (answer, example):
        if not re.fullmatch(r"\d+(,\d+)*;\d+", s or ""):
            return [f"{s!r} is not in the answer format"]
        c, t = s.split(";")
        nums.append(([int(x) for x in c.split(",")], int(t)))
    (ac, at), (ec, et) = nums
    out = []
    if len(ac) == len(ec):
        out.append(f"has as many counts as the gold ({len(ac)})")
    common = sorted((set(ac) | {at}) & (set(ec) | {et}))
    if common:
        out.append(f"shares the number(s) {common} with the gold")
    return out


def lattice_incidence(segs, n):
    """Exact integer incidence: list of sets of lattice points per segment."""
    g = np.arange(n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    out = []
    for (ax, ay), (bx, by) in segs:
        dx, dy = bx - ax, by - ay
        cross = dx * (gy - ay) - dy * (gx - ax)
        dot = (gx - ax) * dx + (gy - ay) * dy
        hit = (cross == 0) & (dot >= 0) & (dot <= dx * dx + dy * dy)
        out.append({(int(x), int(y)) for x, y in zip(gx[hit], gy[hit])})
    return out


def canonical_answer(per_sets):
    union = set().union(*per_sets) if per_sets else set()
    counts = sorted((len(s) for s in per_sets), reverse=True)
    return ",".join(str(c) for c in counts) + ";" + str(len(union)), union


def _capsule_dist(P, A, B):
    """P: (M,2); A,B: (2,) -> distance from each P to segment AB, and projection t."""
    d = B - A
    L2 = float(d @ d)
    t = ((P - A) @ d) / L2
    tc = np.clip(t, 0.0, 1.0)
    F = A + tc[:, None] * d
    return np.hypot(P[:, 0] - F[:, 0], P[:, 1] - F[:, 1]), t


def _seg_seg_dist(a1, a2, b1, b2):
    def p2s(p, a, b):
        d = b - a
        t = max(0.0, min(1.0, float((p - a) @ d) / float(d @ d)))
        return float(np.hypot(*(p - (a + t * d))))
    r = a2 - a1
    s = b2 - b1
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) > 1e-12:
        q = b1 - a1
        t = (q[0] * s[1] - q[1] * s[0]) / den
        u = (q[0] * r[1] - q[1] * r[0]) / den
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            return 0.0
    return min(p2s(a1, b1, b2), p2s(a2, b1, b2), p2s(b1, a1, a2), p2s(b2, a1, a2))


def _angle_deg(u, v):
    a = math.degrees(math.atan2(u[1], u[0])) % 180.0
    b = math.degrees(math.atan2(v[1], v[0])) % 180.0
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line_key(a, b):
    (ax, ay), (bx, by) = a, b
    A, B = by - ay, ax - bx
    C = -(A * ax + B * ay)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) or 1
    k = (A // g, B // g, C // g)
    return max(k, tuple(-x for x in k))


def _overshot(Pa, Pb, ext):
    d = Pb - Pa
    u = d / np.hypot(*d)
    return Pa - u * ext, Pb + u * ext


# ----------------------------------------------------------------------------- pixel decode
def _label(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    return lab, n


def decode_image(image, n_expected, line_rgb):
    """Pixels -> lattice dots + strokes.  Uses meta only for the line colour and N."""
    from scipy import ndimage
    A = np.asarray(image.convert("RGB"), dtype=np.int16)
    H, W = A.shape[:2]
    mx, mn = A.max(axis=2), A.min(axis=2)
    dotmask = (mx < 130) & ((mx - mn) < 45)
    lab, nl = _label(dotmask)
    if nl == 0:
        raise ValueError("no dark components")
    idx = np.arange(1, nl + 1)
    sizes = ndimage.sum(dotmask, lab, idx)
    keep = idx[sizes >= max(4.0, 0.3 * float(np.median(sizes)))]
    if len(keep) != n_expected * n_expected:
        raise ValueError(f"decoded {len(keep)} dots, expected {n_expected}x{n_expected}")
    cent = np.array(ndimage.center_of_mass(dotmask, lab, keep))[:, ::-1]  # (x, y)
    radius = math.sqrt(float(np.median(sizes[keep - 1])) / math.pi)
    x0, x1 = cent[:, 0].min(), cent[:, 0].max()
    y0, y1 = cent[:, 1].min(), cent[:, 1].max()
    N = n_expected
    sp = ((x1 - x0) + (y1 - y0)) / 2.0 / (N - 1)
    gx = np.rint((cent[:, 0] - x0) / sp).astype(int)
    gy = np.rint((cent[:, 1] - y0) / sp).astype(int)
    if gx.min() < 0 or gx.max() != N - 1 or gy.min() < 0 or gy.max() != N - 1:
        raise ValueError("lattice recovery failed")
    pos = np.zeros((N, N, 2))
    filled = np.zeros((N, N), bool)
    for k in range(len(cent)):
        if filled[gx[k], gy[k]]:
            raise ValueError("two dots on one lattice site")
        filled[gx[k], gy[k]] = True
        pos[gx[k], gy[k]] = cent[k]
    lat_res = float(np.max(np.hypot(pos[..., 0] - (x0 + np.arange(N)[:, None] * sp),
                                    pos[..., 1] - (y0 + np.arange(N)[None, :] * sp))))

    lr = np.asarray(line_rgb, float)
    span = float(lr[2] - lr[0])
    strength = np.clip((A[:, :, 2] - A[:, :, 0]) / span, 0.0, 1.0)
    ink = (strength >= INK_FRAC) & ~dotmask
    near = ndimage.binary_dilation(ink, structure=np.ones((3, 3), bool))
    yy, xx = np.mgrid[0:H, 0:W]
    # a dot's disc hides the ink under it: samples within radius+2 carry no evidence
    blocked = np.zeros((H, W), bool)
    rb = radius + 2.0
    ri = int(math.ceil(rb))
    for cx, cy in cent:
        xl, xh = max(0, int(cx) - ri), min(W, int(cx) + ri + 2)
        yl, yh = max(0, int(cy) - ri), min(H, int(cy) + ri + 2)
        blocked[yl:yh, xl:xh] |= ((xx[yl:yh, xl:xh] - cx) ** 2 + (yy[yl:yh, xl:xh] - cy) ** 2) <= rb * rb

    def gather(mask, X, Y):
        xi = np.clip(np.rint(X).astype(int), 0, W - 1)
        yi = np.clip(np.rint(Y).astype(int), 0, H - 1)
        return mask[yi, xi]

    # links between lattice-consecutive dots (gcd of the index step == 1)
    keys = [(i, j) for i in range(N) for j in range(N)]
    G = np.array(keys)
    P = pos.reshape(-1, 2)
    ii, jj = np.triu_indices(len(keys), 1)
    dg = G[jj] - G[ii]
    prim = np.gcd(np.abs(dg[:, 0]), np.abs(dg[:, 1])) == 1
    ii, jj = ii[prim], jj[prim]
    t = np.linspace(0.15, 0.85, 7)[None, :]
    X = P[ii, 0][:, None] * (1 - t) + P[jj, 0][:, None] * t
    Y = P[ii, 1][:, None] * (1 - t) + P[jj, 1][:, None] * t
    ok = (gather(near, X, Y) | gather(blocked, X, Y)).all(axis=1)
    ii, jj = ii[ok], jj[ok]
    links = []
    if len(ii):
        dist = np.hypot(*(P[jj] - P[ii]).T)
        ns = np.maximum(8, np.ceil(dist * 2).astype(int))
        mxn = int(ns.max())
        s = np.arange(mxn)[None, :]
        tt = np.minimum(s / (ns[:, None] - 1), 1.0)
        valid = s < ns[:, None]
        X = P[ii, 0][:, None] * (1 - tt) + P[jj, 0][:, None] * tt
        Y = P[ii, 1][:, None] * (1 - tt) + P[jj, 1][:, None] * tt
        free = valid & ~gather(blocked, X, Y)
        good = (gather(near, X, Y) | ~free).all(axis=1) & (free.sum(axis=1) >= 4)
        links = [(keys[a], keys[b]) for a, b in zip(ii[good], jj[good])]

    # maximal chains along one primitive direction
    nxt = {}
    for a, b in links:
        d = (b[0] - a[0], b[1] - a[1])
        if d[0] < 0 or (d[0] == 0 and d[1] < 0):
            a, b, d = b, a, (-d[0], -d[1])
        nxt[(a, d)] = b
    has_prev = {(b, d) for (a, d), b in nxt.items()}
    runs = []
    for (a, d), b in nxt.items():
        if (a, d) in has_prev:
            continue
        chain = [a]
        cur = a
        while (cur, d) in nxt:
            cur = nxt[(cur, d)]
            chain.append(cur)
        runs.append((d, chain))

    # a stroke visibly continues past both end dots (the 0.35*spacing overshoot)
    strokes, stubs = [], []
    for d, chain in runs:
        pa, pb = pos[chain[0]], pos[chain[-1]]
        u = (pb - pa) / np.hypot(*(pb - pa))
        ss = np.arange(radius + 1.5, 0.28 * sp, 0.5)
        if len(ss) < 2:
            raise ValueError("overshoot probe window empty")
        ends_ok = True
        for p0, sg in ((pb, 1.0), (pa, -1.0)):
            Xs = p0[0] + sg * u[0] * ss
            Ys = p0[1] + sg * u[1] * ss
            if not gather(near, Xs, Ys).all():
                ends_ok = False
        (strokes if ends_ok else stubs).append(chain)
    return dict(N=N, sp=sp, radius=radius, pos=pos, cent=cent, lat_res=lat_res, strength=strength,
                ink=ink, dotmask=dotmask, A=A, strokes=strokes, stubs=stubs, n_links=len(links))


# ----------------------------------------------------------------------------- verify
def analyse(record, image):
    """Returns (problems, stats).  stats are the measured margins (for docs / calibration)."""
    probs: list[str] = []
    st: dict = {}
    mark = [0, "Q"]

    def section(name):
        """Prefix the problems found since the last call with the current layer's letter."""
        for i in range(mark[0], len(probs)):
            if not re.match(r"^[A-Z]: ", probs[i]):
                probs[i] = f"{mark[1]}: {probs[i]}"
        mark[0], mark[1] = len(probs), name
        return probs

    q = record.get("question", "")
    if q != QUESTION:
        probs.append("question text differs from the declared question")
    if not all(ord(c) < 128 for c in q):
        probs.append("question is not pure ASCII")
    for ph in REQUIRED_PHRASES:
        if ph not in q:
            probs.append(f"question does not state {ph!r}")
    if record.get("answer_type") != "text":
        probs.append(f"answer_type {record.get('answer_type')!r} != 'text'")
    if record.get("choices") not in (None, []):
        probs.append("free-form task must not carry choices")

    tier = record.get("tier")
    if tier not in TIERS:
        probs.append(f"T: unknown tier {tier!r}")
        return section(None), st
    T = TIERS[tier]
    meta = record.get("meta") or {}
    try:
        N = int(meta["grid_n"])
        segs = [(tuple(int(v) for v in a), tuple(int(v) for v in b)) for a, b in meta["segments_grid"]]
        R = meta["render"]
        if N < 2 or not isinstance(R, dict):
            raise ValueError(f"grid_n={N}, render={type(R).__name__}")
    except Exception as exc:  # noqa: BLE001 - malformed meta is a problem, not a crash
        probs.append(f"T: meta unreadable: {exc!r}")
        return section(None), st

    # ---------------- T: tier shape and render constants
    section("T")
    L = len(segs)
    if N != T["N"] or L != T["L"] or meta.get("n_segs") != L:
        probs.append(f"tier shape N={N} L={L} != declared N={T['N']} L={T['L']}")
    size = T["size"]
    if list(meta.get("canvas", [])) != [size, size] or image.size != (size, size):
        probs.append(f"canvas {meta.get('canvas')} / image {image.size} != {size}x{size}")
    margin = int(size * MARGIN_CANVAS)
    sp = (size - 2 * margin) / (N - 1)
    dot_r = max(MIN_DOT_DIA / 2.0, T["dot_r_frac"] * sp)
    lw = max(MIN_LW, LW_FRAC * sp)
    for k, v in (("margin_px", margin), ("spacing_px", sp), ("dot_r_px", dot_r), ("line_w_px", lw),
                 ("overshoot_frac", OVERSHOOT)):
        if not isinstance(R.get(k), (int, float)) or abs(R[k] - v) > 1e-9:
            probs.append(f"render.{k}={R.get(k)!r} != tier value {v}")
    mf = meta.get("margin_frac")
    lo, hi = T["band"]
    lo = max(lo, (MIN_CLEAR_PX + dot_r + lw / 2.0) / sp)
    if not isinstance(mf, (int, float)) or not (lo - 5e-4 <= mf <= hi + 5e-4):
        probs.append(f"near-miss margin {mf!r} outside the tier band [{lo:.3f},{hi}]")
        mf = lo

    # ---------------- segment rules (G11, length, range, distinct lines)
    lines = set()
    fatal = False
    for (a, b) in segs:
        if not all(0 <= v < N for v in a + b):
            probs.append(f"segment {a}-{b} leaves the lattice")
            fatal = True
            continue
        dx, dy = b[0] - a[0], b[1] - a[1]
        k = math.gcd(abs(dx), abs(dy))
        if k == 0:
            probs.append(f"degenerate segment {a}-{b}")
            fatal = True
            continue
        p, qq = abs(dx // k), abs(dy // k)
        if p == 0 or qq == 0 or p == qq:
            probs.append(f"G11 trivial slope ({dx // k},{dy // k})")
        if p > MAX_PRIM or qq > MAX_PRIM:
            probs.append(f"direction ({p},{qq}) beyond |p|,|q|<={MAX_PRIM}")
        if k < MIN_STEPS:
            probs.append(f"segment {a}-{b} passes only {k + 1} dots (< {MIN_STEPS + 1})")
        if k * math.hypot(p, qq) < MIN_LEN_FRAC * (N - 1) - 1e-9:
            probs.append(f"segment {a}-{b} shorter than {MIN_LEN_FRAC}*(N-1) spacings")
        lk = _line_key(a, b)
        if lk in lines:
            probs.append(f"U: two segments on one lattice line {lk}")
        lines.add(lk)
    if fatal or not segs:
        if not segs:
            probs.append("no segments")
        return section(None), st

    # ---------------- A: exact answer recompute
    section("A")
    per = lattice_incidence(segs, N)
    gold, union = canonical_answer(per)
    st["answer_recomputed"] = gold
    ans = record.get("answer", "")
    if ans != gold:
        probs.append(f"answer {ans!r} != recomputed {gold!r}")
    if not re.fullmatch(r"\d+(,\d+)*;\d+", ans or ""):
        probs.append("answer does not match 'c1,...,cL;total'")
    else:
        cs = [int(x) for x in ans.split(";")[0].split(",")]
        if len(cs) != L or cs != sorted(cs, reverse=True):
            probs.append("answer counts are not L non-increasing integers")
    m_ex = EXAMPLE_RE.search(q)
    example = m_ex.group(1) if m_ex else EXAMPLE_ANSWER
    st["example"] = example
    for why in example_overlap(ans, example):
        probs.append(f"the question's format example {example!r} {why}")
    if [len(s) for s in per] != list(meta.get("per_line", [])):
        probs.append("meta.per_line disagrees with the lattice scan")
    if {tuple(p) for p in meta.get("on_dots_grid", [])} != union or meta.get("n_on") != len(union):
        probs.append("meta.on_dots_grid / n_on disagree with the lattice scan")
    if len({len(s) for s in per}) < 2:
        probs.append("T: G12 all per-segment counts equal")

    # ---------------- U: geometry in pixels (meta coordinates)
    section("U")
    gidx = np.array([(i, j) for i in range(N) for j in range(N)])
    Ppx = margin + gidx * sp
    over = [_overshot(margin + np.array(a, float) * sp, margin + np.array(b, float) * sp, OVERSHOOT * sp)
            for a, b in segs]
    min_off = min_on = math.inf
    nm_d = np.full(len(gidx), np.inf)
    for s_i, (A0, B0) in enumerate(over):
        d, t = _capsule_dist(Ppx, A0, B0)
        inc = np.array([(int(x), int(y)) in per[s_i] for x, y in gidx])
        if np.any(d[inc] > 1e-6):
            probs.append("an incident dot is off its segment axis")
        clr = d - dot_r - lw / 2.0
        on_mask = np.array([(int(x), int(y)) in union for x, y in gidx])
        sel_off = ~inc & ~on_mask
        sel_on = ~inc & on_mask
        if sel_off.any():
            min_off = min(min_off, float(clr[sel_off].min()))
        if sel_on.any():
            min_on = min(min_on, float(clr[sel_on].min()))
        within = (t >= -1e-9) & (t <= 1 + 1e-9)
        nm_d = np.where(within, np.minimum(nm_d, d), nm_d)
    st["min_off_clear_px"], st["min_on_clear_px"] = round(min_off, 3), round(min_on, 3)
    if min_off < MIN_CLEAR_PX - 1e-6:
        probs.append(f"OFF dot capsule clearance {min_off:.2f}px < {MIN_CLEAR_PX}")
    if min_on < MIN_CLEAR_PX - 1e-6:
        probs.append(f"ON dot clearance to a non-incident segment {min_on:.2f}px < {MIN_CLEAR_PX}")
    off_sel = np.array([(int(x), int(y)) not in union for x, y in gidx])
    near_margin = mf * sp
    n_nm = int(np.sum(off_sel & (nm_d > near_margin) & (nm_d <= NEARMISS_BAND * near_margin)))
    st["n_nearmiss"] = n_nm
    if n_nm < T["nearmiss"]:
        probs.append(f"T: {n_nm} near-miss dots < tier minimum {T['nearmiss']}")
    # the tier's near-miss margin: no OFF dot within margin*spacing of a segment body
    # (meta.margin_frac is rounded to 3 decimals, hence the 5e-4 slack)
    inside = off_sel & (nm_d <= (mf - 5e-4) * sp)
    if inside.any():
        probs.append(f"T: {int(inside.sum())} OFF dot(s) inside the tier's near-miss margin {mf}*spacing")
    min_ep, min_para = math.inf, math.inf
    for i in range(L):
        for j in range(i + 1, L):
            for e1 in over[i]:
                for e2 in over[j]:
                    min_ep = min(min_ep, float(np.hypot(*(e1 - e2))))
            ui, uj = over[i][1] - over[i][0], over[j][1] - over[j][0]
            if _angle_deg(ui, uj) < PARA_ANG_DEG:
                min_para = min(min_para, _seg_seg_dist(over[i][0], over[i][1], over[j][0], over[j][1]))
    st["min_endpoint_sep_sp"] = round(min_ep / sp, 3)
    if min_ep < EP_SEP_FRAC * sp - 1e-6:
        probs.append(f"G7 overshot endpoints {min_ep:.2f}px apart < {EP_SEP_FRAC}*spacing")
    if min_para < PARA_SEP_FRAC * sp - 1e-6:
        probs.append(f"G8 near-parallel segments {min_para:.2f}px apart < {PARA_SEP_FRAC}*spacing")

    # ---------------- K: core_key
    section("K")
    if record.get("core_key") != expected_core_key(segs, N):
        probs.append("core_key is not the (D4 x translation)-canonical key of the segment set")

    # ---------------- R: the PNG is the documented rendering of meta
    section("R")
    try:
        probs += _render_layer(image, N, size, segs, R, sp, dot_r, lw, margin, st)
    except Exception as exc:  # noqa: BLE001
        probs.append(f"re-render failed: {exc!r}")

    # ---------------- P: pixel decode
    section("P")
    try:
        probs += _pixel_layer(image, N, segs, per, union, R, sp, dot_r, lw, margin, st)
    except (ValueError, IndexError, ZeroDivisionError, FloatingPointError) as exc:
        probs.append(f"pixel decode failed: {exc!r}")
    return section(None), st


def render_meta(N, size, segs, sp, dot_r, lw, margin, line_rgb, dot_rgb=(35, 35, 35), bg=(255, 255, 255),
                overshoot=OVERSHOOT, ss=3):
    """The rendering the record documents (meta.render), implemented here from its description:
    segments overshot by overshoot*spacing at both ends, then all dots on top, at ss x
    supersampling, LANCZOS-downsampled."""
    from PIL import Image, ImageDraw
    big = Image.new("RGB", (size * ss, size * ss), tuple(bg))
    d = ImageDraw.Draw(big)
    for a, b in segs:
        A0, B0 = _overshot(margin + np.array(a, float) * sp, margin + np.array(b, float) * sp, overshoot * sp)
        d.line([(float(A0[0]) * ss, float(A0[1]) * ss), (float(B0[0]) * ss, float(B0[1]) * ss)],
               fill=tuple(line_rgb), width=max(1, int(round(lw * ss))))
    rr = dot_r * ss
    for gx in range(N):
        for gy in range(N):
            cx, cy = (margin + gx * sp) * ss, (margin + gy * sp) * ss
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=tuple(dot_rgb))
    return big.resize((size, size), Image.LANCZOS)


def stray_marks(img, N, segs, sp, dot_r, lw, margin):
    """Pixels darker than white by > STRAY_DEV that lie neither on a declared dot (dot_r + pad of
    a lattice point) nor on a declared overshot stroke (lw/2 + pad of its axis).  Returns
    (count of stray pixels, max deviation from white over every pixel outside those regions)."""
    dev = 255 - img.min(axis=2).astype(np.int16)
    ys, xs = np.nonzero(dev > 4)
    if not len(xs):
        return 0, 0
    Q = np.stack([xs, ys], 1).astype(float) + 0.5            # pixel centres in drawing coordinates
    g = np.clip(np.rint((Q - margin) / sp), 0, N - 1)
    covered = np.hypot(*(Q - (margin + g * sp)).T) <= dot_r + STRAY_DOT_PAD
    for a, b in segs:
        A0, B0 = _overshot(margin + np.array(a, float) * sp, margin + np.array(b, float) * sp, OVERSHOOT * sp)
        d, _ = _capsule_dist(Q, A0, B0)
        covered |= d <= lw / 2.0 + STRAY_STROKE_PAD
    out = dev[ys, xs][~covered]
    return int((out > STRAY_DEV).sum()), int(out.max()) if len(out) else 0


def _render_layer(image, N, size, segs, R, sp, dot_r, lw, margin, st):
    import PIL
    probs = []
    for k, v in (("bg_rgb", [255, 255, 255]), ("dot_rgb", [35, 35, 35])):
        if list(R.get(k, [])) != v:
            probs.append(f"render.{k}={R.get(k)!r} != {v}")
    lr = list(R.get("line_rgb", []))
    if len(lr) != 3 or lr[2] - lr[0] < 100 or R.get("line_color") != "blue":
        probs.append(f"line colour {R.get('line_color')!r} {lr} is not the blue the question names")
        return probs
    img = np.asarray(image.convert("RGB"), dtype=np.int16)
    # marks away from every declared dot and stroke (independent of the Pillow version)
    n_stray, max_out = stray_marks(img, N, segs, sp, dot_r, lw, margin)
    st["stray_px"], st["max_offstroke_dev"] = n_stray, max_out
    if n_stray:
        probs.append(f"{n_stray} pixel(s) darker than white by > {STRAY_DEV} away from every declared dot "
                     f"and stroke (max {max_out})")
    ref = np.asarray(render_meta(N, size, segs, sp, dot_r, lw, margin, lr), dtype=np.int16)
    if ref.shape != img.shape:
        return probs + [f"image shape {img.shape} != rendering {ref.shape}"]
    diff = np.abs(ref - img).max(axis=2)
    exact = PIL.__version__ in EXACT_PILLOW
    st["rerender_mode"] = "exact" if exact else f"tolerant (Pillow {PIL.__version__})"
    st["rerender_max_diff"] = int(diff.max())
    st["rerender_frac_gt16"] = round(float((diff > 16).mean()), 5)
    if exact:
        if diff.max() > RERENDER_EXACT_MAX:
            probs.append(f"PNG differs from the rendering of meta (max channel diff {int(diff.max())} > "
                         f"{RERENDER_EXACT_MAX} under Pillow {PIL.__version__}; {int((diff > 0).sum())} px differ)")
    elif diff.max() > RERENDER_MAX or (diff > 16).mean() > RERENDER_FRAC:
        probs.append(f"PNG differs from the rendering of meta (max {int(diff.max())}, "
                     f"{100 * float((diff > 16).mean()):.2f}% of pixels > 16)")
    return probs


def _pixel_layer(image, N, segs, per, union, R, sp, dot_r, lw, margin, st):
    probs = []
    dec = decode_image(image, N, R.get("line_rgb", (32, 140, 220)))
    pos = dec["pos"]
    # dots where the declared lattice says (pixel-index centroids sit RENDER_OFFSET from draw coords)
    want = margin + np.stack(np.meshgrid(np.arange(N), np.arange(N), indexing="ij"), -1) * sp + RENDER_OFFSET
    pos_err = float(np.max(np.hypot(*(pos - want).reshape(-1, 2).T)))
    st["dot_pos_err_px"] = round(pos_err, 3)
    if pos_err > PIX_POS_TOL:
        probs.append(f"decoded dots {pos_err:.2f}px from the declared lattice")
    st["decoded_dot_r_px"] = round(dec["radius"], 2)
    if abs(dec["radius"] - dot_r) > 1.0:
        probs.append(f"decoded dot radius {dec['radius']:.2f} != declared {dot_r:.2f}")

    # decoded strokes == declared segments
    dec_sets = {frozenset(c) for c in dec["strokes"]}
    meta_sets = {frozenset(s) for s in per}
    st["decoded_strokes"] = len(dec_sets)
    st["stubs"] = len(dec["stubs"])
    if dec_sets != meta_sets:
        miss = len(meta_sets - dec_sets)
        extra = len(dec_sets - meta_sets)
        probs.append(f"image strokes != meta segments ({miss} missing, {extra} extra)")
    else:
        dec_answer, _ = canonical_answer([set(c) for c in dec["strokes"]])
        st["answer_from_pixels"] = dec_answer
    long_stubs = [c for c in dec["stubs"] if len(c) >= 3]
    if long_stubs:
        probs.append(f"{len(long_stubs)} ink chain(s) of >=3 dots without visible overshoot")

    # stroke geometry in image coordinates, from the decoded dot centroids
    strokes = []
    for s_i, (a, b) in enumerate(segs):
        pa, pb = pos[a], pos[b]
        A0, B0 = _overshot(pa, pb, OVERSHOOT * dec["sp"])
        strokes.append((A0, B0))

    # every ink pixel lies on some stroke
    ys, xs = np.nonzero(dec["ink"])
    Q = np.stack([xs, ys], 1).astype(float)
    dmin = np.full(len(Q), np.inf)
    owner = np.full(len(Q), -1)
    for s_i, (A0, B0) in enumerate(strokes):
        d, _ = _capsule_dist(Q, A0, B0)
        better = d < dmin
        owner[better] = s_i
        dmin = np.minimum(dmin, d)
    worst = float(dmin.max()) if len(dmin) else 0.0
    st["max_ink_dist_px"] = round(worst - lw / 2.0, 3)
    if worst > lw / 2.0 + EXTRA_INK_TOL:
        probs.append(f"ink {worst - lw / 2.0:.2f}px outside every stroke")

    # the ink axis of each stroke passes through the centroids of its ON dots
    strength = dec["strength"]
    worst_axis = 0.0
    for s_i, (A0, B0) in enumerate(strokes):
        u = (B0 - A0) / np.hypot(*(B0 - A0))
        sel = (owner == s_i) & (dmin <= lw / 2.0 + 1.5)
        # drop pixels near other strokes (crossings); dot pixels are not in the ink mask
        for o_i, (A1, B1) in enumerate(strokes):
            if o_i != s_i:
                d, _ = _capsule_dist(Q, A1, B1)
                sel &= d > lw + 2.0
        Qs = Q[sel]
        if len(Qs) < 10:
            probs.append(f"stroke {s_i} has too little free ink to fit an axis")
            continue
        w = strength[Qs[:, 1].astype(int), Qs[:, 0].astype(int)]
        c = (Qs * w[:, None]).sum(0) / w.sum()
        Cm = ((Qs - c) * w[:, None]).T @ (Qs - c)
        _evals, evecs = np.linalg.eigh(Cm)
        axis_u = evecs[:, 1]
        axis_n = np.array([-axis_u[1], axis_u[0]])
        if abs(float(axis_u @ u)) < math.cos(math.radians(1.0)):
            probs.append(f"stroke {s_i} ink axis deviates from the lattice direction")
        for g in per[s_i]:
            off = abs(float((want[g] - c) @ axis_n))
            worst_axis = max(worst_axis, off)
    st["max_axis_off_px"] = round(worst_axis, 3)
    if worst_axis > AXIS_TOL:
        probs.append(f"a stroke's ink axis misses an ON dot centre by {worst_axis:.2f}px")

    # visible separation between every non-incident (dot, stroke) pair closer than 1.2 spacings.
    # Walk from the dot centre to the nearest point of the stroke axis; after the last dot
    # pixel, count the samples before the first pixel of THIS stroke's ink (blue, within
    # lw/2+1px of its axis).  Ink of a third stroke on the way (e.g. a stroke through this
    # very dot) counts as separation, it is not this stroke.  For OFF dots (on no stroke)
    # also measure the pure-white run (every channel >= WHITE_MIN_CH), as gate G4 does.
    A = dec["A"]
    white = (A.min(axis=2) >= WHITE_MIN_CH)
    dark = dec["dotmask"]
    H, W = white.shape
    flat = pos.reshape(-1, 2)
    gl = [(i, j) for i in range(N) for j in range(N)]
    min_gap = min_white = math.inf
    n_pairs = 0
    for s_i, (A0, B0) in enumerate(strokes):
        d, t = _capsule_dist(flat, A0, B0)
        dd = B0 - A0
        for k in np.nonzero(d < 1.2 * dec["sp"])[0]:
            if gl[k] in per[s_i]:
                continue
            p = flat[k]
            tc = min(1.0, max(0.0, float((p - A0) @ dd) / float(dd @ dd)))
            f = A0 + tc * dd
            dist = float(np.hypot(*(f - p)))
            ss = np.arange(0.0, dist + 0.25, 0.25)
            S = p[None, :] + (f - p)[None, :] * (np.minimum(ss, dist) / dist)[:, None]
            X = np.clip(np.rint(S[:, 0]).astype(int), 0, W - 1)
            Y = np.clip(np.rint(S[:, 1]).astype(int), 0, H - 1)
            ds, _ = _capsule_dist(S, A0, B0)
            own = (strength[Y, X] >= INK_FRAC) & (ds <= lw / 2.0 + 1.0)
            dk = dark[Y, X]
            first_own = int(np.argmax(own)) if own.any() else len(ss)
            last_dark = int(np.nonzero(dk[:first_own])[0].max()) if dk[:first_own].any() else -1
            seg = slice(last_dark + 1, first_own)
            gap = 0.25 * float((~dk[seg]).sum())
            min_gap = min(min_gap, gap)
            n_pairs += 1
            if gl[k] not in union:
                min_white = min(min_white, 0.25 * float(white[Y[seg], X[seg]].sum()))
    st["gap_pairs"] = n_pairs
    st["min_gap_px"] = None if math.isinf(min_gap) else round(min_gap, 2)
    st["min_off_white_px"] = None if math.isinf(min_white) else round(min_white, 2)
    if min_gap < MIN_GAP_PX:
        probs.append(f"only {min_gap:.2f}px between a dot and a stroke not through it (< {MIN_GAP_PX})")
    if min_white < MIN_WHITE_PX:
        probs.append(f"OFF dot white gap {min_white:.2f}px (< {MIN_WHITE_PX})")
    return probs


def verify(record, image):
    return analyse(record, image)[0]
