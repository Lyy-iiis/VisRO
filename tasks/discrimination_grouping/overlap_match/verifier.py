"""Independent verifier for overlap_match.

Imports nothing from the generator (tasks/discrimination_grouping/overlap_match/generator/*, generator/base,
the adapter).  Inputs: the record (question, answer, choices, meta) and the
decoded PNG.  The layout constants below restate the delivery layout; they are
not read from the generator.

  V0  contract      question text, choices A-D, answer, meta.answer, image size,
                    tier structure (n shapes, H same-base hosts, one target); the
                    question shows no example answer letter
  V1  panels        image -> structure: each A-D cell holds exactly its recorded
                    candidate (every outline point on ink, every ink pixel on the
                    outline), fully inside its frame with clearance, clear of the
                    letter; the letters read A, B, C, D; candidates pairwise
                    distinct
  V2  stack         image -> structure: the stack ink is exactly the recorded
                    stack outlines (none missing, nothing extra), none clipped,
                    no ink in the 2 px gutter before the divider
  V3  gold          the answer's candidate equals the recorded target shape
                    vertex for vertex after translation; no other stack shape
                    is a translate of any candidate; its outline lies on the
                    stack ink at that offset
  V4  uniqueness,   for every OTHER candidate, at every translation on a 0.5 px
      ink lattice   lattice, at least one outline point (snapped to a pixel) is
                    >= GAP_MIN px from all stack ink pixels.  This is the SAME
                    definition as the generator's gap gate (gen_tiers.gap_gate),
                    re-implemented: a consistency check of the release floor,
                    not an independent method.
  V5  uniqueness,   a different method: exact point-to-segment distance to the
      geometry      recorded stack CENTRELINES (V2 ties them to the pixels), over
                    EVERY real translation (a lattice screen with a Lipschitz
                    slack, then an exact branch-and-bound on the cells the screen
                    leaves open).  At every real translation, every other
                    candidate keeps a point >= CL_MIN (3.5) px from every
                    centreline, and >= CL_ARC_MIN (6) px of its outline at
                    >= CL_ARC_TH (2.5) px, where its stroke would stick out of
                    the drawn ink (a visible misfit, not a single point).
  V6  uniqueness,   a third method that reads no meta: each candidate's ink is
      pixels only   cut from its cell and slid over the stack's non-white pixels
                    (grey < 250, antialiased fringe included) at every integer
                    translation.  The record's answer must be the only candidate
                    that fits (all of its pixels within PX_FIT), and every other
                    candidate keeps a pixel >= PX_FAR from any stack pixel.

Pixel convention (measured, see docs): the PNG is drawn at 2x and LANCZOS-halved,
so a recorded coordinate x lands on pixel index x - 0.5 (pixel i covers [i, i+1)).
"""
from __future__ import annotations

import re

import numpy as np

QUESTION = (
    'Exactly one of the shapes A-D on the right appears in the overlapping outlines on the left, '
    'at the same size and orientation (moved only, never rotated or resized). Which one? Answer '
    'with the letter.'
)
LETTERS = "ABCD"
CHOICE_RANGE = "A-D"     # names the whole choice set; it is not an example answer

# documented layout
W, H = 896, 504
STACK_W = 392              # stack panel = columns [0, 392); the divider is grey, not ink
GUTTER = 2                 # the last 2 stack columns (390, 391) must hold no ink (checked in V2)
CELL = 252
PAD = 8
FRAME = 234                # rounded frame side; frame line 2 px drawn inward
FRAME_LINE = 2
STROKE = 3.0
INK = 170                  # grey < 170 is ink
LETTER_PIX = (8, 10, 21, 24)   # glyph ink: pixel indices relative to the frame origin
LETTER_BOX = (8.0, 10.0, 22.0, 25.0)   # the same area in coordinates (pixel i covers [i, i+1))

# tolerances (calibrated on the 3 px LANCZOS-antialiased stroke; see docs)
ON_INK = 1.5               # a recorded outline point must be within this of ink
OFF_INK = 2.5              # an ink pixel must be within this of a recorded outline
CLEAR = 2.0                # stroke-to-frame and stroke-to-letter white, px
GAP_MIN = 3.5              # V4: a wrong candidate keeps a point this far from ink (0.5 px lattice)
PIX = 0.5                  # recorded coordinate x -> pixel index x - PIX
PHASES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))   # sub-pixel phases of the gap scan
DISTINCT = 6.0             # min pairwise Hausdorff between candidates (registered frame)
STACK_MARGIN = 3.0         # stack outlines stay this far inside the stack panel

# V5: exact centreline geometry over every real translation
CL_H = 0.5                 # grid spacing of the exact centreline-distance field, px
CL_PAD = 8.0               # the field extends this far outside the stack panel
CL_CAP = 8.0               # larger distances are stored as CL_CAP (only values near CL_MIN matter)
CL_SLACK = CL_H * 2 ** 0.5     # sample snapped to the grid (<= h/sqrt2) + real t inside a lattice cell (<= h/sqrt2)
CL_MIN = 3.5               # every wrong outline keeps a point >= this from every stack centreline
                           # (two 3 px strokes 3.5 px apart leave 0.5 px of white between them)
CL_ARC_TH = 2.5            # a 3 px stroke this far off every drawn centreline sticks out of the drawn ink
CL_ARC_MIN = 6.0           # px of every wrong outline at >= CL_ARC_TH, at every real translation
CL_DENSE = 0.1             # centreline sample spacing of the exact refinement (KD error <= CL_DENSE / 2)
CL_MAX_CELLS = 400         # lattice cells the exact refinement may open per candidate

# V6: pixels only
PX_LENIENT = 250           # V6 stack pixels: anything not white (antialiased fringe included)
PX_PAD = 24                # white margin around the stack panel for the V6 scan
PX_FIT = 1.5               # the gold's pixels all lie within this of a stack pixel at some integer shift
PX_FAR = 2.5               # every other candidate keeps a pixel >= this from all stack pixels at every shift

CORRECTNESS = ("Decodes the PNG: the stack ink must be exactly the recorded stack outlines (every outline point "
               "within 1.5 px of ink, every ink pixel within 2.5 px of an outline, no ink in the gutter) and each "
               "A-D cell exactly its recorded candidate; the gold is recomputed as the candidate that equals the "
               "recorded target shape vertex for vertex after translation, and its outline must lie within 1.5 px of "
               "stack ink there.  Independently of meta, the candidate pixels cut from the cells are slid over the "
               "stack pixels and the answer must be the only candidate that fits.")
UNIQUENESS = (f"Three methods, each must pass for every other candidate.  V4 (the generator's gap-gate definition, "
              f"re-implemented): at every translation on a 0.5 px lattice some outline point is >= {GAP_MIN} px from "
              f"all stack ink pixels.  V5 (exact centreline geometry, every real translation, certified): some "
              f"outline point stays >= {CL_MIN} px from every recorded stack centreline, and >= {CL_ARC_MIN} px of "
              f"outline stays >= {CL_ARC_TH} px away (a visible misfit, not a single point).  V6 "
              f"(pixels only, no meta): the candidate's pixels cut from its cell keep a pixel >= {PX_FAR} px from "
              f"every non-white stack pixel at every integer shift, while the gold fits within {PX_FIT} px.  Also: "
              f"no other stack shape is a translate of a candidate, candidates are pairwise >= 6 px apart, each fits "
              f"its frame clear of the letter, and the letters read A, B, C, D.")

TIER_SHAPE = {"easy": (6, 1), "medium": (8, 1), "hard": (9, 2), "xhard": (10, 3), "xxhard": (11, 4)}


# ------------------------------------------------------------------ helpers
def _samples(poly, step=0.5) -> np.ndarray:
    P = np.asarray(poly, float)
    Q = np.roll(P, -1, axis=0)
    out = []
    for a, b in zip(P, Q):
        k = max(2, int(np.hypot(*(b - a)) / step))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    return np.concatenate(out)


def _seg_dist(pts: np.ndarray, polys) -> np.ndarray:
    """Exact distance from each point to the nearest segment of any polygon."""
    A = np.concatenate([np.asarray(p, float) for p in polys])
    B = np.concatenate([np.roll(np.asarray(p, float), -1, axis=0) for p in polys])
    d = B - A
    dd = np.maximum((d * d).sum(1), 1e-12)
    best = np.full(len(pts), np.inf)
    for s in range(0, len(pts), 4096):
        P = pts[s:s + 4096]
        w = P[:, None, :] - A[None]
        t = np.clip((w * d[None]).sum(-1) / dd[None], 0.0, 1.0)
        q = A[None] + t[..., None] * d[None]
        best[s:s + 4096] = np.sqrt(((P[:, None, :] - q) ** 2).sum(-1)).min(1)
    return best


def _edt(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~mask)


def _at(D: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """D at rounded pixel positions; outside the map counts as infinitely far."""
    xi = np.rint(pts[:, 0]).astype(int)
    yi = np.rint(pts[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < D.shape[1]) & (yi >= 0) & (yi < D.shape[0])
    out = np.full(len(pts), np.inf)
    out[ok] = D[yi[ok], xi[ok]]
    return out


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max()))


def every_translation_leaves_a_gap(D: np.ndarray, pts: np.ndarray, gap: float) -> bool:
    """True iff for EVERY integer translation t that keeps all rounded points
    inside D, some point p has D[p + t] >= gap.  (A translation that pushes a
    point outside the stack panel trivially leaves that point far from ink.)"""
    from scipy.signal import fftconvolve
    xi = np.rint(pts[:, 0]).astype(int)
    yi = np.rint(pts[:, 1]).astype(int)
    xi -= xi.min()
    yi -= yi.min()
    if yi.max() >= D.shape[0] or xi.max() >= D.shape[1]:
        return True
    K = np.zeros((yi.max() + 1, xi.max() + 1))
    np.add.at(K, (yi, xi), 1.0)
    far = (D >= gap).astype(float)
    counts = fftconvolve(far, K[::-1, ::-1], mode="valid")
    return float(counts.min()) >= 0.5


# --------------------------------------------- V5: exact centreline geometry
def _dense(poly, step) -> np.ndarray:
    """Outline points no more than `step` apart (ceil per edge), vertices included."""
    P = np.asarray(poly, float)
    out = []
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        k = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    return np.concatenate(out)


def centreline_field(polys, h=CL_H, pad=CL_PAD, cap=CL_CAP) -> np.ndarray:
    """Exact distance (capped) from every point of a grid of spacing h over
    [-pad, STACK_W + pad] x [-pad, H + pad] to the nearest segment of any
    polygon.  Grid index (j, i) is the point (i*h - pad, j*h - pad)."""
    nx = int(round((STACK_W + 2 * pad) / h)) + 1
    ny = int(round((H + 2 * pad) / h)) + 1
    G = np.full((ny, nx), float(cap))
    for P in polys:
        P = np.asarray(P, float)
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            lo, hi = np.minimum(a, b) - cap, np.maximum(a, b) + cap
            i0, i1 = max(0, int(np.floor((lo[0] + pad) / h))), min(nx, int(np.ceil((hi[0] + pad) / h)) + 1)
            j0, j1 = max(0, int(np.floor((lo[1] + pad) / h))), min(ny, int(np.ceil((hi[1] + pad) / h)) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            tx = (np.arange(i0, i1) * h - pad - a[0])[None, :]
            ty = (np.arange(j0, j1) * h - pad - a[1])[:, None]
            d = b - a
            u = np.clip((tx * d[0] + ty * d[1]) / max(float(d @ d), 1e-12), 0.0, 1.0)
            np.minimum(G[j0:j1, i0:i1], np.hypot(tx - u * d[0], ty - u * d[1]), out=G[j0:j1, i0:i1])
    return G


class _Corr:
    """Valid-mode correlation of binary fields with templates, sharing FFTs:
    out[j, i] = sum_{v,u} field[j+v, i+u] * K[v, u]."""

    def __init__(self, shape, kshapes):
        from scipy import fft
        self.fft = fft
        self.shape = shape
        self.s = tuple(fft.next_fast_len(shape[d] + max(k[d] for k in kshapes) - 1, True) for d in (0, 1))

    def field(self, M):
        return self.fft.rfft2(M.astype(np.float64), self.s)

    def kernel(self, K):
        return self.fft.rfft2(K[::-1, ::-1], self.s)

    def valid(self, Ff, Kf, kshape):
        full = self.fft.irfft2(Ff * Kf, self.s)
        return full[kshape[0] - 1:self.shape[0], kshape[1] - 1:self.shape[1]]


def _template(S, h):
    q = np.rint(S / h).astype(int)
    q0 = q.min(0)
    q = q - q0
    K = np.zeros((q[:, 1].max() + 1, q[:, 0].max() + 1))
    np.add.at(K, (q[:, 1], q[:, 0]), 1.0)
    return K, q0


def _certify_cell(tree, S, t, r, th, err=CL_DENSE / 2):
    """Best-first branch and bound over the real translations c with
    |c - t|_inf <= r.  f(c) = max_p dist(p + c, centrelines) is 1-Lipschitz in
    c, and the KD distance to the dense centreline samples overestimates the
    exact one by <= err, so a cell of half-side rr is certified when
    kd(c) - err - rr*sqrt2 >= th.  Returns (True, None) if f >= th on the whole
    cell, else (False, value): value < th is a real translation where every
    sampled outline point is within th of a centreline; th <= value means the
    cell could not be certified at 0.01 px resolution."""
    import heapq

    def node(c, rr):
        kd = float(tree.query(S + c, distance_upper_bound=th + 2 * rr + 2 * err + 1.0)[0].max())
        return (kd - err - rr * 2 ** 0.5, kd, rr, tuple(c))

    heap = [node(np.asarray(t, float), float(r))]
    while heap:
        lb, kd, rr, c = heapq.heappop(heap)
        if lb >= th:
            return True, None
        if kd < th or rr < 0.01:
            return False, kd
        q = rr / 2
        for dx in (-q, q):
            for dy in (-q, q):
                heapq.heappush(heap, node(np.asarray(c) + [dx, dy], q))
    return True, None


def centreline_uniqueness(polys, cands, gold_index):
    """V5.  For each wrong candidate: (min_ok, worst, arc_px).  min_ok: at EVERY
    real translation some outline point is >= CL_MIN from every centreline.
    arc_px: the least outline length (px) that stays >= CL_ARC_TH from every
    centreline, over every real translation (certified lower bound)."""
    from scipy.spatial import cKDTree
    G = centreline_field(polys)
    tree = cKDTree(np.concatenate([_dense(p, CL_DENSE) for p in polys]))
    wrong = [i for i in range(len(cands)) if i != gold_index]
    S = {i: _dense(cands[i], 0.5) for i in wrong}
    T = {i: _template(S[i], CL_H) for i in wrong}
    C = _Corr(G.shape, [T[i][0].shape for i in wrong])
    F_min = C.field(G >= CL_MIN + CL_SLACK)
    F_arc = C.field(G >= CL_ARC_TH + CL_SLACK)
    out = {}
    for i in wrong:
        K, q0 = T[i]
        Kf = C.kernel(K)
        cnt = C.valid(F_min, Kf, K.shape)
        open_cells = np.argwhere(cnt < 0.5)
        ok, worst = True, None
        if len(open_cells) > CL_MAX_CELLS:
            ok, worst = False, float("nan")
        for jy, jx in open_cells if ok or worst is None else ():
            t = np.array([(jx - q0[0]) * CL_H - CL_PAD, (jy - q0[1]) * CL_H - CL_PAD])
            good, f = _certify_cell(tree, S[i], t, CL_H / 2, CL_MIN)
            if not good:
                ok, worst = False, f if worst is None else min(worst, f)
        per = float(np.hypot(*(np.roll(np.asarray(cands[i], float), -1, 0) - np.asarray(cands[i], float)).T).sum())
        arc = float(C.valid(F_arc, Kf, K.shape).min()) * per / len(S[i])
        out[i] = (ok, worst, arc)
    return out


# --------------------------------------------------------- V6: pixels only
def _cell_pixels(allink, i) -> np.ndarray:
    """The candidate's ink pixels cut from cell i (the letter glyph removed)."""
    r, c = divmod(i, 2)
    x0, y0 = STACK_W + c * CELL, r * CELL
    ink = allink[y0:y0 + CELL, x0:x0 + CELL].copy()
    ink[PAD + LETTER_PIX[1] - 1:PAD + LETTER_PIX[3] + 2, PAD + LETTER_PIX[0] - 1:PAD + LETTER_PIX[2] + 2] = False
    ys, xs = np.nonzero(ink)
    return np.stack([xs, ys], 1)


def pixel_match(gray) -> list[tuple[bool, bool]]:
    """V6.  For each cell: (fits, far).  fits: at some integer shift every
    candidate pixel is within PX_FIT of a non-white stack pixel.  far: at every
    integer shift some candidate pixel is >= PX_FAR from all of them."""
    from scipy.ndimage import distance_transform_edt
    allink = gray < INK
    st = np.zeros((H + 2 * PX_PAD, STACK_W + 2 * PX_PAD), bool)
    st[PX_PAD:PX_PAD + H, PX_PAD:PX_PAD + STACK_W - GUTTER] = gray[:, :STACK_W - GUTTER] < PX_LENIENT
    D = distance_transform_edt(~st)
    Ks = []
    for i in range(4):
        P = _cell_pixels(allink, i)
        if not len(P):
            Ks.append(None)
            continue
        q = P - P.min(0)
        K = np.zeros((q[:, 1].max() + 1, q[:, 0].max() + 1))
        K[q[:, 1], q[:, 0]] = 1.0
        Ks.append(K)
    C = _Corr(D.shape, [K.shape for K in Ks if K is not None] or [(1, 1)])
    F_fit, F_far = C.field(D > PX_FIT), C.field(D >= PX_FAR)
    out = []
    for K in Ks:
        if K is None:
            out.append((False, True))
            continue
        Kf = C.kernel(K)
        out.append((float(C.valid(F_fit, Kf, K.shape).min()) < 0.5, float(C.valid(F_far, Kf, K.shape).min()) >= 0.5))
    return out


def format_example_letters(question: str) -> list[str]:
    """Letters the question shows as an example answer: every standalone A-D
    token outside the choice-range marker "A-D"."""
    return re.findall(r"(?<![A-Za-z])[A-D](?![A-Za-z])", question.replace(CHOICE_RANGE, " "))


_GLYPHS = None


def _glyphs():
    """Reference A-D glyph masks, rendered the documented way (Pillow default
    font at 40 px on a 2x canvas, LANCZOS down), in frame coords [0,30)^2."""
    global _GLYPHS
    if _GLYPHS is None:
        from PIL import Image, ImageDraw, ImageFont
        f = ImageFont.load_default(40)
        out = {}
        for L in LETTERS:
            im = Image.new("RGB", (60, 60), "white")
            ImageDraw.Draw(im).text((16, 10), L, fill=(40, 40, 44), font=f)
            out[L] = np.asarray(im.resize((30, 30), Image.LANCZOS).convert("L")) < INK
        _GLYPHS = out
    return _GLYPHS


# ------------------------------------------------------------------- verify
def verify(record: dict, image) -> list[str]:
    try:
        return _verify(record, image)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return [f"meta malformed: {exc!r}"]


def _verify(record: dict, image) -> list[str]:
    probs: list[str] = []
    meta = record["meta"]
    ans = record.get("answer")

    # ---------------------------------------------------------------- V0
    if record.get("question") != QUESTION:
        probs.append("V0 question text is not the canonical overlap_match question")
    shown = format_example_letters(str(record.get("question", "")))
    if shown:
        probs.append(f"V0 the question shows example answer letter(s) {shown}"
                     + (" equal to the gold" if ans in shown else ""))
    if record.get("answer_type") != "choice" or list(record.get("choices") or []) != list(LETTERS):
        probs.append("V0 answer_type/choices must be choice over A-D")
    if ans not in LETTERS:
        return probs + [f"V0 answer {ans!r} not a letter A-D"]
    gi = LETTERS.index(ans)
    if meta.get("answer") != ans or meta.get("gold_index") != gi:
        probs.append("V0 meta answer/gold_index disagree with the record answer")
    if image.size != (W, H):
        return probs + [f"V0 image is {image.size}, expected {(W, H)}"]
    st, cd = meta["stack"], meta["candidates"]
    polys = [np.asarray(p, float) for p in st["polys"]]
    kinds = list(st["kinds"])
    tk = int(st["target_k"])
    cands = [np.asarray(p, float) for p in cd["polys"]]
    anchors = [np.asarray(a, float) for a in cd["anchors"]]
    if len(cands) != 4 or len(anchors) != 4:
        return probs + ["V0 need exactly four candidates"]
    n_want, h_want = TIER_SHAPE[record["tier"]]
    if len(polys) != n_want or len(kinds) != n_want:
        probs.append(f"V0 tier {record['tier']} needs {n_want} stack shapes, meta has {len(polys)}")
    if kinds.count("target") != 1 or not (0 <= tk < len(kinds)) or kinds[tk] != "target":
        probs.append("V0 exactly one stack shape must be the target, at target_k")
    if kinds.count("host") != h_want:
        probs.append(f"V0 tier {record['tier']} needs {h_want} same-base hosts, meta has {kinds.count('host')}")
    if probs:
        return probs

    gray = np.asarray(image.convert("L"))
    allink = gray < INK

    # ---------------------------------------------------------------- V1
    letter_ok = []
    for i in range(4):
        r, c = divmod(i, 2)
        fx, fy = STACK_W + c * CELL + PAD, r * CELL + PAD          # frame origin
        if np.abs(anchors[i] - (np.asarray([fx, fy]) + np.asarray(meta["layout"]["panel_off"], float))).max() > 1e-3:
            probs.append(f"V1 cell {LETTERS[i]}: anchor is not frame origin + panel_off")
        x0, y0 = STACK_W + c * CELL, r * CELL                       # cell origin
        ink = allink[y0:y0 + CELL, x0:x0 + CELL].copy()
        # the glyph's ink lies inside LETTER_BOX; a candidate is kept >= 3.5 px
        # (stroke/2 + CLEAR) from that box, so its ink never enters it
        lx0, ly0 = fx - x0 + LETTER_PIX[0], fy - y0 + LETTER_PIX[1]
        lx1, ly1 = fx - x0 + LETTER_PIX[2], fy - y0 + LETTER_PIX[3]
        letter = np.zeros_like(ink)
        letter[ly0:ly1 + 1, lx0:lx1 + 1] = ink[ly0:ly1 + 1, lx0:lx1 + 1]
        outline = ink & ~letter
        S = _samples(cands[i] + anchors[i]) - [x0, y0]
        # fits the frame with clearance (geometry)
        lo = fx - x0 + FRAME_LINE + STROKE / 2 + CLEAR - 1e-3
        hi = fx - x0 + FRAME - FRAME_LINE - STROKE / 2 - CLEAR + 1e-3
        loy, hiy = lo - (fx - x0) + (fy - y0), hi - (fx - x0) + (fy - y0)
        if S[:, 0].min() < lo or S[:, 0].max() > hi or S[:, 1].min() < loy or S[:, 1].max() > hiy:
            probs.append(f"V1 cell {LETTERS[i]}: candidate outline leaves its frame (clearance {CLEAR} px)")
        # clears the letter (geometry)
        bx0, by0 = fx - x0 + LETTER_BOX[0], fy - y0 + LETTER_BOX[1]
        bx1, by1 = fx - x0 + LETTER_BOX[2], fy - y0 + LETTER_BOX[3]
        dx = np.maximum(np.maximum(bx0 - S[:, 0], S[:, 0] - bx1), 0)
        dy = np.maximum(np.maximum(by0 - S[:, 1], S[:, 1] - by1), 0)
        if np.hypot(dx, dy).min() < STROKE / 2 + CLEAR - 1e-3:
            probs.append(f"V1 cell {LETTERS[i]}: candidate outline runs into the letter")
        # every outline point on ink; every non-letter ink pixel on the outline
        if not outline.any():
            probs.append(f"V1 cell {LETTERS[i]}: no candidate ink")
            continue
        d_on = _at(_edt(outline), S - PIX)
        if d_on.max() > ON_INK:
            probs.append(f"V1 cell {LETTERS[i]}: recorded candidate not drawn (max {d_on.max():.2f} px off ink)")
        ys, xs = np.nonzero(outline)
        d_off = _seg_dist(np.stack([xs, ys], 1) + PIX, [cands[i] + anchors[i] - [x0, y0]])
        if d_off.max() > OFF_INK:
            probs.append(f"V1 cell {LETTERS[i]}: ink not on the recorded candidate (max {d_off.max():.2f} px)")
        # outline ink strictly inside the frame line
        fr0, fr1 = fx - x0 + FRAME_LINE, fx - x0 + FRAME - FRAME_LINE
        fry0, fry1 = fy - y0 + FRAME_LINE, fy - y0 + FRAME - FRAME_LINE
        if xs.min() < fr0 or xs.max() >= fr1 or ys.min() < fry0 or ys.max() >= fry1:
            probs.append(f"V1 cell {LETTERS[i]}: candidate ink touches or crosses the frame")
        # the letter: separated from the outline and reading the right letter
        if letter.any():
            gap = _at(_edt(letter), np.stack([xs, ys], 1).astype(float)).min()
            if gap < 1.5:
                probs.append(f"V1 cell {LETTERS[i]}: candidate ink touches the letter ({gap:.1f} px)")
        crop = letter[ly0:ly1 + 1, lx0:lx1 + 1]
        win = (slice(LETTER_PIX[1], LETTER_PIX[3] + 1), slice(LETTER_PIX[0], LETTER_PIX[2] + 1))
        iou = {L: (g[win] & crop).sum() / max(1, (g[win] | crop).sum()) for L, g in _glyphs().items()}
        best = max(iou, key=iou.get)
        letter_ok.append(best == LETTERS[i] and iou[best] >= 0.8)
    if len(letter_ok) == 4 and not all(letter_ok):
        probs.append("V1 the cell letters do not read A, B, C, D (top-left, top-right, bottom-left, bottom-right)")
    E = [_samples(c, 1.0) for c in cands]
    for i in range(4):
        for j in range(i + 1, 4):
            h = _hausdorff(E[i], E[j])
            if h < DISTINCT:
                probs.append(f"V1 candidates {LETTERS[i]} and {LETTERS[j]} are only {h:.1f} px apart")

    # ---------------------------------------------------------------- V2
    sink = allink[:, :STACK_W].copy()
    if sink[:, STACK_W - GUTTER:].any():
        probs.append(f"V2 ink in the stack gutter (columns {STACK_W - GUTTER}-{STACK_W - 1}): "
                     f"{int(sink[:, STACK_W - GUTTER:].sum())} px")
    Ds = _edt(sink)
    for k, p in enumerate(polys):
        S = _samples(p)
        if (S[:, 0].min() < STACK_MARGIN or S[:, 1].min() < STACK_MARGIN or
                S[:, 0].max() > STACK_W - GUTTER - STACK_MARGIN or S[:, 1].max() > H - STACK_MARGIN):
            probs.append(f"V2 stack shape {k} is clipped by the stack panel edge")
        dk = _at(Ds, S - PIX)
        if dk.max() > ON_INK:
            probs.append(f"V2 stack shape {k} ({kinds[k]}) is not drawn (max {dk.max():.2f} px off ink)")
    ys, xs = np.nonzero(sink)
    d_off = _seg_dist(np.stack([xs, ys], 1) + PIX, polys)
    if d_off.max() > OFF_INK:
        probs.append(f"V2 stack ink not explained by the recorded shapes (max {d_off.max():.2f} px, "
                     f"{int((d_off > OFF_INK).sum())} px)")

    # ---------------------------------------------------------------- V3
    gold = cands[gi]
    pos = np.asarray(st["pos"][tk], float)
    if gold.shape != polys[tk].shape or np.abs(gold + pos - polys[tk]).max() > 0.01:
        probs.append(f"V3 candidate {ans} is not the recorded target shape translated")
    for k, p in enumerate(polys):
        pc = p - (p.min(0) + p.max(0)) / 2
        for i, c in enumerate(cands):
            if c.shape != p.shape or (k == tk and i == gi):
                continue
            cc = c - (c.min(0) + c.max(0)) / 2
            if np.abs(cc - pc).max() < 1.0:
                probs.append(f"V3 stack shape {k} is a copy of candidate {LETTERS[i]}")
    dg = _at(Ds, _samples(gold + pos) - PIX)
    if dg.max() > ON_INK:
        probs.append(f"V3 gold outline leaves the stack ink by {dg.max():.2f} px at its offset")

    # ---------------------------------------------------------------- V4
    for i, c in enumerate(cands):
        if i == gi:
            continue
        S = _samples(c, 0.5) - PIX
        for ph in PHASES:
            if not every_translation_leaves_a_gap(Ds, S + np.asarray(ph), GAP_MIN):
                probs.append(f"V4 candidate {LETTERS[i]} can be laid on the stack with every outline point "
                             f"< {GAP_MIN} px from ink (phase {ph}): not provably absent")
                break

    # ---------------------------------------------------------------- V5
    for i, (ok, worst, arc) in centreline_uniqueness(polys, cands, gi).items():
        if not ok:
            probs.append(f"V5 candidate {LETTERS[i]} is not certified to keep an outline point >= {CL_MIN} px from "
                         f"every stack centreline at every real translation (closest found: {worst:.3f} px)")
        if arc < CL_ARC_MIN:
            probs.append(f"V5-arc candidate {LETTERS[i]}: at some translation only {arc:.1f} px of its outline is "
                         f">= {CL_ARC_TH} px from every stack centreline (need {CL_ARC_MIN})")

    # ---------------------------------------------------------------- V6
    pm = pixel_match(gray)
    fits = [LETTERS[i] for i, (f, _) in enumerate(pm) if f]
    if fits != [ans]:
        probs.append(f"V6 pixel-only matching: the candidates that fit the stack are {fits}, expected only {ans}")
    for i, (_, far) in enumerate(pm):
        if i != gi and not far:
            probs.append(f"V6 candidate {LETTERS[i]}'s pixels come within {PX_FAR} px of stack pixels everywhere "
                         f"at some shift")
    return probs
