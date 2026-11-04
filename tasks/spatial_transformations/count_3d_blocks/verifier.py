"""Independent verifier for count_3d_blocks.

Nothing here imports the generator (generator, shared/drawing/isometric.py,
extra_difficulty.py).  The isometric projection, the painter's-order renderer,
the camera rule, the shading and the robust-pixel rule
are re-implemented from their written specification; byte equality of the
re-render with the delivered PNG ties this implementation to the data.

Pipeline for one record
  1. form      answer is a plain integer, answer_type int, question equals the
               declared template for the tier's variant (total / minimum) and
               contains no digit (no format example that could match a gold).
     member    meta.palette is one of the tier's release greys; meta.heightmap
               fits the tier (TIER_SPEC: base shape, max height, the seeded 2x2
               hidden block of medium/hard, easy's 4-14 cubes, the answer and
               occlusion floors) and meta.base_rows/base_cols/occlusion agree.
               These are reported without stopping the pixel checks below.
  2. render    camera re-derived from meta.heightmap (margin 46, 2x supersample)
               must equal meta.placement; re-rendering heightmap + palette + frame
               must give the delivered image byte for byte.
  3. witness   meta.minimum_heightmap W <= H, sum(W) == answer, and W renders
               byte-identically to H (an explicit structure with `answer` cubes
               that looks exactly like the image).
  4. decode    face types (background/top/left/right) are read from the PNG
               colours alone; a pixel counts only when it and its 8 neighbours
               have the same exact class ("robust").
  5. ILP       0/1 variable per cube position on the drawn (N+1)x(M+1) grid (the
               question says cubes stand on its squares), levels 0..max(H)+1, gravity
               x[k+1] <= x[k].  At every robust pixel the front-most cube that
               touches it must cover it fully with the decoded face type, and
               robust background stays uncovered.  Pixels with identical
               candidate lists form one group; a group may be "violated" at a cost
               equal to its pixel count.  Budget 0 (pixel-exact): the minimum
               cube count is the answer (for 'total' questions the maximum too).
               Margin: no structure with fewer (for 'total': or more) cubes
               violates <= MARGIN_PX-1 pixels -- solved as "minimise violated
               pixels subject to count <= answer-1", which must be infeasible.
               So every different count disagrees with the image on >=
               MARGIN_PX robust pixels.  W itself must violate nothing (checks
               the rasterizer against the real image).
  6. percept   each forced top cube of W owns >= CORE_PX robust interior pixels.
"""
from __future__ import annotations

import os

# The integer programs and numpy are single-threaded workloads; OpenBLAS's spinning
# worker threads otherwise multiply CPU use by ~3x per process on a shared host.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

CORRECTNESS = (
    "The PNG is re-rendered byte-for-byte from meta.heightmap with an independent isometric rasterizer (camera re-derived, "
    "palette from meta and pinned to the tier's release greys); the stored witness renders identically and sums to the "
    "answer; an integer program over every gravity-supported cube set on the drawn grid, constrained by face types decoded "
    "from the PNG colours, proves no consistent structure has fewer cubes; and the heightmap fits the declared tier.")
UNIQUENESS = (
    "One integer is asked: the integer program shows any cube structure on the drawn grid with fewer cubes (and, for 'how "
    "many' questions, more cubes) disagrees with at least 100 robust pixels of the image, and every forced top cube shows "
    "at least 12 robust interior pixels.")

# ---- written specification of the renderer (not imported) -----------------
SS = 2                      # supersampling factor
HW0, HH0, CH0 = 34, 17, 40  # half tile width, half tile height, cube height (px)
MARGIN = 46                 # placement margin (px at final resolution)
EDGE = (20, 20, 20)
FLOOR = (200, 205, 212)
FRAME = (225, 225, 230)
SHADE = (1.0, 0.74, 0.88)   # top, left (+j face), right (+i face)
SIZES = {"easy": 448, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}
VARIANT = {"easy": "total", "medium": "minimum", "hard": "minimum", "xhard": "minimum", "xxhard": "minimum"}

_PREFIX = "The image shows a figure built from identical unit cubes standing on the squares of the grid drawn on the floor."
QUESTIONS = {
    "total": (
        'How many unit cubes make up this figure? Every cube rests on the floor grid or on '
        'another cube. Answer with an integer.'
    ),
    "minimum": (
        'This figure of unit cubes stands on the floor grid; no cube floats, some may be hidden, '
        'and the cubes need not be connected. What is the fewest cubes that would look exactly '
        'like this from this viewpoint? Answer with an integer.'
    ),
}

# Release palettes (written spec of the generator's base greys).  Pinned so a record re-rendered in a
# low-contrast colour cannot pass: the robust-pixel gates count colour classes, not contrast.
_GREYS = {(205, 208, 214), (208, 211, 216), (200, 203, 210)}
PALETTES = {"easy": _GREYS, "medium": _GREYS, "hard": _GREYS, "xhard": {(205, 208, 214)}, "xxhard": {(205, 208, 214)}}

# Tier membership (written spec of TIER_DEFINITIONS), checked on meta.heightmap H and the answer:
#   shape: allowed (rows, cols); hmax: max(H) must equal it (medium/hard: the seeded front L) or not exceed it;
#   block: (back heights, front height) of the seeded 2x2 hidden block; total: allowed sum(H) for easy;
#   min_answer / min_occlusion: lower bounds on the answer and on sum(H) - answer.
TIER_SPEC = {
    "easy": dict(shape={(3, 4), (4, 4)}, hmax=2, hmax_exact=False, block=None, total=(4, 14), min_answer=1,
                 min_occlusion=0),
    "medium": dict(shape={(r, c) for r in (3, 4) for c in (3, 4)}, hmax=3, hmax_exact=True, block=({1}, 3),
                   total=None, min_answer=1, min_occlusion=1),
    "hard": dict(shape={(r, c) for r in (3, 4) for c in (3, 4)}, hmax=4, hmax_exact=True, block=({1, 2}, 4),
                 total=None, min_answer=1, min_occlusion=1),
    "xhard": dict(shape={(4, 5)}, hmax=5, hmax_exact=False, block=None, total=None, min_answer=30, min_occlusion=2),
    "xxhard": dict(shape={(5, 5)}, hmax=6, hmax_exact=False, block=None, total=None, min_answer=45, min_occlusion=4),
}

# Robust pixels any different-count structure must change.  The projection is not true isometric:
# cube (i,j,k) and cube (i+1,j+1,k+1) are drawn 6 px apart, so a cheaper look-alike differs from the image
# only where an edge moves by 6 px.  Exact margins of 120 release-sample items (30 per minimum tier): a
# whole top face moved costs 112+ robust px; a partly hidden face moved costs 34-91 px, a sliver a solver
# cannot be expected to read.  100 keeps the first and rejects the second.
MARGIN_PX = 100
CORE_PX = 12     # robust interior pixels of each forced top cube
COLOR_TOL = 1    # exact palette match (+-1); the floor line colour is 2 levels from one palette's top shade
EXTRA_LEVELS = 2  # ILP height cap = max(H) + EXTRA_LEVELS


def shade(base, f):
    return tuple(max(0, min(255, int(c * f))) for c in base)


def _bbox(H, hw, hh, ch):
    N, M = len(H), len(H[0])
    pts = [(0, 0, 0), (M + 1, 0, 0), (0, N + 1, 0), (M + 1, N + 1, 0)]
    for j in range(N):
        for i in range(M):
            if H[j][i] > 0:
                for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    pts += [(i + di, j + dj, H[j][i]), (i + di, j + dj, 0)]
    xs = [(i - j) * hw for i, j, k in pts]
    ys = [(i + j) * hh - k * ch for i, j, k in pts]
    return min(xs), min(ys), max(xs), max(ys)


def camera(H, size):
    """Placement rule: fit floor (+1 pad) and cubes into size-2*46 px, centred, at 2x."""
    hw, hh, ch = HW0 * SS, HH0 * SS, CH0 * SS
    x0, y0, x1, y1 = _bbox(H, hw, hh, ch)
    canvas = size * SS
    scale = min(1.0, (canvas - 2 * MARGIN * SS) / max(x1 - x0, y1 - y0, 1))
    hw, hh, ch = max(2, int(hw * scale)), max(1, int(hh * scale)), max(2, int(ch * scale))
    x0, y0, x1, y1 = _bbox(H, hw, hh, ch)
    return dict(hw=hw, hh=hh, ch=ch, origin=((canvas - (x1 - x0)) / 2 - x0, (canvas - (y1 - y0)) / 2 - y0), ss=SS)


def _pt(cam, i, j, k, dx=0.0, dy=0.0):
    ox, oy = cam["origin"]
    return (ox + (i - j) * cam["hw"] - dx, oy + (i + j) * cam["hh"] - k * cam["ch"] - dy)


def faces(cam, i, j, k, dx=0.0, dy=0.0):
    """(top, left, right) polygons of cube (i, j, k) at supersampled resolution."""
    P = lambda a, b, c: _pt(cam, i + a, j + b, k + c, dx, dy)  # noqa: E731
    top = [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]
    left = [P(0, 1, 1), P(1, 1, 1), P(1, 1, 0), P(0, 1, 0)]
    right = [P(1, 0, 1), P(1, 1, 1), P(1, 1, 0), P(1, 0, 0)]
    return top, left, right


def painter_cells(N, M):
    """Back-to-front: ascending i+j (stable over row-major order), then level."""
    return sorted(((i, j) for j in range(N) for i in range(M)), key=lambda c: c[0] + c[1])


def render(H, cam, size, base, frame=True):
    N, M = len(H), len(H[0])
    im = Image.new("RGB", (size * SS, size * SS), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(M + 2):
        d.line([_pt(cam, i, 0, 0), _pt(cam, i, N + 1, 0)], fill=FLOOR, width=SS)
    for j in range(N + 2):
        d.line([_pt(cam, 0, j, 0), _pt(cam, M + 1, j, 0)], fill=FLOOR, width=SS)
    cols = [shade(base, f) for f in SHADE]
    for i, j in painter_cells(N, M):
        for k in range(H[j][i]):
            polys = faces(cam, i, j, k)
            for poly, col in zip(polys, cols):
                d.polygon(poly, fill=col, outline=EDGE)
            for poly in polys:
                d.line(poly + [poly[0]], fill=EDGE, width=SS)
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    if frame:
        ImageDraw.Draw(im).rounded_rectangle([3, 3, size - 4, size - 4], radius=14, outline=FRAME, width=2)
    return im


# ---- pixel decoding ----------------------------------------------------------
def _robust(ids):
    """Keep a label only where its 3x3 neighbourhood carries the same label."""
    n0, n1 = ids.shape
    pad = np.pad(ids, 1, constant_values=-99)
    keep = np.ones(ids.shape, bool)
    for dy in range(3):
        for dx in range(3):
            keep &= pad[dy:dy + n0, dx:dx + n1] == ids
    return keep


def decode_faces(image, base):
    """-1 background, 0 top, 1 left, 2 right, -2 not robust; from PNG colours only."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    cls = np.full(a.shape[:2], -2, np.int32)
    refs = [(-1, (255, 255, 255))] + [(f, shade(base, s)) for f, s in enumerate(SHADE)]
    for c, col in refs:
        cls[np.abs(a - np.array(col, np.int16)).max(axis=2) <= COLOR_TOL] = c
    return np.where(_robust(cls), cls, -2)


def footprint(cam, i, j, k, size):
    """Final-resolution pixels a cube paints: 1/2/3 = fully covered by face top/left/right
    (no edge line inside), 9 = touched otherwise.  Returns (y0, x0, array) or None."""
    polys = faces(cam, i, j, k)
    xs = [p[0] for q in polys for p in q]
    ys = [p[1] for q in polys for p in q]
    x0, x1 = max(0, int(min(xs) // SS) - 2), min(size, int(max(xs) // SS) + 3)
    y0, y1 = max(0, int(min(ys) // SS) - 2), min(size, int(max(ys) // SS) + 3)
    if x1 <= x0 or y1 <= y0:
        return None
    im = Image.new("L", ((x1 - x0) * SS, (y1 - y0) * SS), 0)
    d = ImageDraw.Draw(im)
    local = faces(cam, i, j, k, dx=x0 * SS, dy=y0 * SS)
    for f, poly in enumerate(local):
        d.polygon(poly, fill=1 + f, outline=200)
    for poly in local:
        d.line(poly + [poly[0]], fill=200, width=SS)
    a = np.asarray(im).reshape(y1 - y0, SS, x1 - x0, SS).transpose(0, 2, 1, 3)
    lo, hi = a.min(axis=(2, 3)), a.max(axis=(2, 3))
    out = np.zeros(lo.shape, np.uint8)
    out[hi > 0] = 9
    full = (lo == hi) & (lo >= 1) & (lo <= 3)
    out[full] = lo[full]
    return y0, x0, out


def cube_cores(W, cam, size):
    """Robust interior final pixels owned by each cube of W (cube-ID rendering)."""
    N, M = len(W), len(W[0])
    im = Image.new("I", (size * SS, size * SS), 0)
    d = ImageDraw.Draw(im)
    ids, n = {}, 0
    for i, j in painter_cells(N, M):
        for k in range(W[j][i]):
            n += 1
            ids[i, j, k] = n
            polys = faces(cam, i, j, k)
            for poly in polys:
                d.polygon(poly, fill=n, outline=0)
            for poly in polys:
                d.line(poly + [poly[0]], fill=0, width=SS)
    a = np.asarray(im).reshape(size, SS, size, SS).transpose(0, 2, 1, 3)
    lo, hi = a.min(axis=(2, 3)), a.max(axis=(2, 3))
    lab = np.where(lo == hi, lo, 0)
    lab = np.where(_robust(lab), lab, 0)
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    return {c: int(counts[v]) for c, v in ids.items()}


# ---- integer program -----------------------------------------------------------
def build_program(ref, cam, size, N, M, K):
    """Variables for every cell of the drawn (N+1) x (M+1) floor grid, levels 0..K-1."""
    cells = [(i, j) for j in range(N + 1) for i in range(M + 1)]
    var = {}
    for i, j in cells:
        for k in range(K):
            var[i, j, k] = len(var)
    nv = len(var)
    order = sorted(var, key=lambda c: (c[0] + c[1], c[1], c[2]))  # painter order (ties never overlap)
    rank = np.empty(nv, np.int64)
    for r, c in enumerate(order):
        rank[var[c]] = r
    P, V, T = [], [], []
    useful = ref != -2
    for c in order:
        fp = footprint(cam, *c, size)
        if fp is None:
            continue
        y0, x0, out = fp
        yy, xx = np.nonzero(out)
        gy, gx = yy + y0, xx + x0
        keep = useful[gy, gx]
        P.append(gy[keep] * size + gx[keep])
        V.append(np.full(int(keep.sum()), var[c], np.int64))
        T.append(out[yy, xx][keep])
    P, V, T = np.concatenate(P), np.concatenate(V), np.concatenate(T)
    srt = np.lexsort((rank[V], P))
    P, V, T = P[srt], V[srt], T[srt]
    groups = {}
    starts = np.flatnonzero(np.r_[True, P[1:] != P[:-1]]) if len(P) else np.array([], np.int64)
    ends = np.r_[starts[1:], len(P)]
    covered = np.zeros(size * size, bool)
    for s, e in zip(starts, ends):
        p = int(P[s])
        covered[p] = True
        sig = (int(ref.flat[p]), tuple(zip(V[s:e].tolist(), T[s:e].tolist())))
        groups[sig] = groups.get(sig, 0) + 1
    # robust face pixels that no candidate cube can even touch: unexplainable
    orphan = int(((ref >= 0).ravel() & ~covered).sum())
    return var, groups, orphan


def _program(var, groups, K, budget, count_bound=None):
    """Constraint matrix over [cube vars | group slacks].  A group (pixels with identical candidate lists)
    is either satisfied or violated as a whole, so its slack costs its pixel count; total slack cost
    <= budget.  count_bound = (lo, hi) optionally bounds the number of cubes."""
    nv, ng = len(var), len(groups)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, lb, ub):
        nonlocal r
        for c, v in entries:
            rows.append(r); cols.append(c); vals.append(v)
        lo.append(lb); hi.append(ub)
        r += 1

    upper = np.ones(nv + ng)
    cost = np.zeros(nv + ng)
    for g, ((t, cand), w) in enumerate(groups.items()):
        s = nv + g
        cost[s] = float(w)
        if w > budget:
            upper[s] = 0  # this group alone would exceed the budget
        if t == -1:  # background: no cube may touch it
            for v, _ in cand:
                add([(v, 1.0), (s, -1.0)], -np.inf, 0.0)
            continue
        good = [n for n, (v, ft) in enumerate(cand) if ft == t + 1]
        add([(cand[n][0], 1.0) for n in good] + [(s, 1.0)], 1.0, np.inf)  # some full face of type t
        for n, (v, ft) in enumerate(cand):  # any other toucher must lie behind a present good one
            if ft == t + 1:
                continue
            ahead = [cand[m][0] for m in good if m > n]
            add([(v, 1.0)] + [(a, -1.0) for a in ahead] + [(s, -1.0)], -np.inf, 0.0)
    for (i, j, k), v in var.items():  # support from directly below
        if k + 1 < K:
            add([(var[i, j, k + 1], 1.0), (v, -1.0)], -np.inf, 0.0)
    if ng:
        add([(s, cost[s]) for s in range(nv, nv + ng)], -np.inf, float(budget))
    if count_bound is not None:
        add([(v, 1.0) for v in range(nv)], *count_bound)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv + ng))
    return LinearConstraint(A, lo, hi), Bounds(0, upper), cost


def extreme_count(var, groups, K, budget, maximize=False):
    """Fewest (or most) cubes over gravity-supported structures on the grid that violate at most
    `budget` robust pixels of the decoded image.  Returns the cube count, or None if the program fails."""
    nv, ng = len(var), len(groups)
    cons, bounds, _ = _program(var, groups, K, budget)
    c = np.zeros(nv + ng)
    c[:nv] = -1.0 if maximize else 1.0
    res = milp(c, constraints=cons, integrality=np.ones(nv + ng), bounds=bounds)
    if res.status != 0:
        return None
    return int(round(res.x[:nv].sum()))


def lookalike(var, groups, K, budget, count_bound):
    """A structure whose cube count lies in count_bound and that violates at most `budget` robust pixels,
    found by minimising the violated pixels (much faster to refute than minimising the count at that
    budget, and equivalent: none exists iff the count extreme at `budget` stays outside count_bound).
    Returns ("none", None, None), ("found", cubes, violated pixels) or ("failed", status, None)."""
    nv, ng = len(var), len(groups)
    cons, bounds, cost = _program(var, groups, K, budget, count_bound)
    res = milp(cost, constraints=cons, integrality=np.ones(nv + ng), bounds=bounds)
    if res.status == 2:  # infeasible
        return "none", None, None
    if res.status != 0:
        return "failed", res.status, None
    return "found", int(round(res.x[:nv].sum())), int(round(res.fun))


def violations(var, groups, X):
    """Robust pixels violated by an explicit heightmap X (indexed X[j][i], cells of the drawn grid)."""
    present = set()
    for (i, j, k), v in var.items():
        if 0 <= j < len(X) and 0 <= i < len(X[0]) and k < X[j][i]:
            present.add(v)
    bad = 0
    for (t, cand), w in groups.items():
        front = None
        for v, ft in cand:  # back -> front
            if v in present:
                front = ft
        ok = (front is None) if t == -1 else (front == t + 1)
        bad += 0 if ok else w
    return bad


# ---- the verifier ----------------------------------------------------------------
def tier_problems(tier, H, gold, meta):
    """The record belongs to its declared tier (TIER_SPEC) and its size fields are consistent."""
    sp = TIER_SPEC[tier]
    N, M = len(H), len(H[0])
    mx, tot = max(map(max, H)), sum(map(sum, H))
    p = []
    if (N, M) not in sp["shape"]:
        p.append(f"base {N}x{M} is not a {tier} shape {sorted(sp['shape'])}")
    if mx > sp["hmax"] or (sp["hmax_exact"] and mx != sp["hmax"]):
        p.append(f"max height {mx} does not fit {tier} (cap {sp['hmax']}{', exact' if sp['hmax_exact'] else ''})")
    if sp["block"]:
        back, front = sp["block"]
        if not any(H[j][i] in back and H[j][i + 1] == H[j + 1][i] == H[j + 1][i + 1] == front
                   for j in range(N - 1) for i in range(M - 1)):
            p.append(f"no seeded 2x2 hidden block (back in {sorted(back)}, front {front}) in the {tier} heightmap")
    if sp["total"] and not sp["total"][0] <= tot <= sp["total"][1]:
        p.append(f"{tot} cubes is outside the {tier} range {sp['total']}")
    if gold < sp["min_answer"]:
        p.append(f"answer {gold} < {sp['min_answer']} for {tier}")
    if tot - gold < sp["min_occlusion"]:
        p.append(f"occlusion {tot - gold} < {sp['min_occlusion']} for {tier}")
    if (meta.get("base_rows"), meta.get("base_cols")) != (N, M):
        p.append(f"meta.base_rows/base_cols {meta.get('base_rows')}x{meta.get('base_cols')} != heightmap {N}x{M}")
    if meta.get("occlusion") != tot - gold:
        p.append(f"meta.occlusion {meta.get('occlusion')} != total - answer {tot - gold}")
    return p


def _grid(x, name):
    if not isinstance(x, list) or not x or not all(isinstance(r, list) and len(r) == len(x[0]) for r in x):
        raise ValueError(f"{name} is not a rectangular list of lists")
    if not all(isinstance(v, int) and 0 <= v <= 20 for r in x for v in r):
        raise ValueError(f"{name} has a non-integer or out-of-range height")
    return x


def verify(record, image, *, details=None):
    """Problems with the record (empty == pass).  Tier-membership and palette problems are collected
    separately so they never stop the pixel checks from running."""
    member = []
    return _verify(record, image, details, member) + member


def _verify(record, image, details, member):
    probs = []
    m = record.get("meta", {})
    tier = record.get("tier")
    ans = record.get("answer")
    if tier not in VARIANT:
        return [f"unknown tier {tier!r}"]
    variant = VARIANT[tier]
    # 1. form
    if not isinstance(ans, str) or not re.fullmatch(r"0|[1-9][0-9]*", ans):
        return [f"answer {ans!r} is not a plain integer"]
    gold = int(ans)
    if record.get("answer_type") != "int":
        probs.append(f"answer_type {record.get('answer_type')!r} != 'int'")
    q = record.get("question")
    if q != QUESTIONS[variant]:
        probs.append(f"question text is not the declared {variant!r} template")
    if not isinstance(q, str) or re.search(r"[0-9]", q):
        # no format example: a number in the question could equal or share digits with the answer
        probs.append("question contains a digit (format examples are not allowed for this task)")
    if m.get("variant") != variant:
        probs.append(f"meta.variant {m.get('variant')!r} != {variant!r} for tier {tier}")
    try:
        H = _grid(m.get("heightmap"), "heightmap")
        W = _grid(m.get("minimum_heightmap"), "minimum_heightmap")
        base = tuple(int(c) for c in m["palette"])
        size = int(m["resolution"])
    except (KeyError, TypeError, ValueError) as exc:
        return probs + [f"meta unusable: {exc}"]
    N, M = len(H), len(H[0])
    if base not in PALETTES[tier]:
        member.append(f"meta.palette {list(base)} is not a {tier} release palette {sorted(PALETTES[tier])}")
    member += tier_problems(tier, H, gold, m)
    if size != SIZES[tier] or image.size != (size, size):
        probs.append(f"resolution {image.size} / meta {size} != tier size {SIZES[tier]}")
        return probs
    if (len(W), len(W[0])) != (N, M) or any(W[j][i] > H[j][i] for j in range(N) for i in range(M)):
        return probs + ["minimum_heightmap is not a sub-structure of heightmap"]
    if sum(map(sum, W)) != gold:
        probs.append(f"witness has {sum(map(sum, W))} cubes but the answer is {gold}")
    if variant == "total" and W != H:
        probs.append("'how many' question but the witness differs from the heightmap")
    if m.get("total_cubes") != sum(map(sum, H)) or m.get("minimum_cubes") != gold:
        probs.append("meta total/minimum counts disagree with heightmap/answer")
    # 2. camera + byte-exact re-render
    cam = camera(H, size)
    pl = m.get("placement") or {}
    if (pl.get("hw"), pl.get("hh"), pl.get("ch"), pl.get("ss")) != (cam["hw"], cam["hh"], cam["ch"], SS) or \
            [float(v) for v in pl.get("origin", ())] != list(cam["origin"]):
        probs.append(f"meta.placement {pl} != camera re-derived from the heightmap {cam}")
    img = image.convert("RGB")
    if render(H, cam, size, base).tobytes() != img.tobytes():
        probs.append("image is not the byte-exact rendering of meta.heightmap")
        return probs  # nothing below is meaningful against an unexplained image
    # 3. witness reproduces the image
    if W != H and render(W, cam, size, base).tobytes() != img.tobytes():
        probs.append("witness heightmap does not render identically to the image")
    if probs:
        return probs
    # 4-5. decode faces from pixels; integer program
    ref = decode_faces(img, base)
    K = max(map(max, H)) + EXTRA_LEVELS
    var, groups, orphan = build_program(ref, cam, size, N, M, K)
    if orphan:
        probs.append(f"{orphan} robust face pixels cannot be explained by any cube position")
        return probs
    vw = violations(var, groups, W)
    if vw:
        probs.append(f"the witness violates {vw} robust pixels of the decoded image (decode/raster mismatch)")
        return probs
    lb0 = extreme_count(var, groups, K, 0)
    info = {"min_exact": lb0, "groups": len(groups), "robust_face_px": int((ref >= 0).sum())}
    if lb0 is None or lb0 < gold:
        probs.append(f"a structure with {lb0} < {gold} cubes matches every robust pixel of the image")
    else:
        st, n, px = lookalike(var, groups, K, MARGIN_PX - 1, (0, gold - 1))
        info["fewer_lookalike"] = (n, px) if st == "found" else st
        if st == "found":
            probs.append(f"a structure with {n} < {gold} cubes differs from the image on fewer than {MARGIN_PX} robust "
                         f"pixels ({px})")
        elif st == "failed":
            probs.append(f"margin program failed (status {n})")
    if variant == "total":
        ub0 = extreme_count(var, groups, K, 0, maximize=True)
        info["max_exact"] = ub0
        if ub0 is None or ub0 > gold:
            probs.append(f"a structure with {ub0} > {gold} cubes matches every robust pixel of the image")
        else:
            st, n, px = lookalike(var, groups, K, MARGIN_PX - 1, (gold + 1, np.inf))
            info["more_lookalike"] = (n, px) if st == "found" else st
            if st == "found":
                probs.append(f"a structure with {n} > {gold} cubes differs from the image on fewer than {MARGIN_PX} "
                             f"robust pixels ({px})")
            elif st == "failed":
                probs.append(f"margin program failed (status {n})")
    # 6. every forced top is clearly visible
    cores = cube_cores(W, cam, size)
    weak = [(i, j, W[j][i], cores[i, j, W[j][i] - 1]) for j in range(N) for i in range(M)
            if W[j][i] and cores[i, j, W[j][i] - 1] < CORE_PX]
    info["min_top_core"] = min((cores[i, j, W[j][i] - 1] for j in range(N) for i in range(M) if W[j][i]), default=None)
    if weak:
        probs.append(f"forced top cube(s) with < {CORE_PX} robust interior pixels: {weak[:3]}")
    if details is not None:
        details.update(info)
    return probs
