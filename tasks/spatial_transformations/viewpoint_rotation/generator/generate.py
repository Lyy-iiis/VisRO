"""Generator of viewpoint_rotation (mental rotation).

LEVEL = high (3D mental simulation). A reference isometric cube-stack is shown
on top; four option panels (A-D) below show candidate structures rendered with
the SAME projection. Exactly one option is the reference structure seen from a
different direction (the camera yawed by 90/180/270 degrees about the vertical
axis); the others are mirror images or structures with 1-2 columns edited.

Ground truth is exact by construction
-------------------------------------
Yawing the camera by 90 degrees * k around a gravity-filled heightmap H and
re-rendering with this fixed isometric projection produces the same image as
rendering the ROTATED heightmap rot90(H, k). So the correct option is simply a
render of rot90(H, k) -- no perceptual labeling step.

Ambiguity guards (all exact):
* H must be chiral: its mirror must not equal any of its 4 rotations
  (canonical-form check), otherwise a mirror distractor would be a right answer.
* H must have no nontrivial rotational self-symmetry, so the correct option
  never looks identical to the reference.
* Every option heightmap is deduped by canonical form (min over its 4 rotations
  of (shape, bytes)), so no distractor is any rotation of the truth and no two
  options are rotations of each other.
* PIXEL guard against occlusion ambiguity: every distractor's render must
  differ from the renders of ALL four rotations of H (a distractor that only
  differs in a fully hidden column would otherwise be visually identical to
  the correct answer). All four option renders must be pairwise distinct too.
* REFERENCE guard (every rung; xhard.py calls it for xhard/xxhard too): the pixel guard above only looks at the option side.  A column hidden
  in the REFERENCE panel can take other heights without any visible change,
  and a distractor that is a yaw of such a variant would be a second correct
  answer.  ReferenceGuard finds
  those hidden heights at the delivered reference size and refuses any
  distractor that is a yaw of a reference-consistent variant or not readably
  different (the difference3d rule) from a view of one at the delivered
  option size, or closer than half a cube top face of readable area to an
  aligned view of H, or closer than half a cube top face of FACE-LEVEL
  difference (outline pixels excluded, face_level_px) to any view of any
  reference-consistent variant, H included.  It resamples references whose panel allows an
  unreadable change of two cells that single-column changes cannot explain
  (e.g. a column one cell nearer along the view diagonal and one cube taller,
  whose top lands 6 base px from the old one).  xhard/xxhard additionally
  require every column top of the reference to show >= 8 interior pixels
  (extra.readable); the release verifier
  (tasks/spatial_transformations/viewpoint_rotation/verifier.py) re-checks all of this
  independently.

Difficulty ladder (meta.rung; extra_difficulty.base_rung maps easy->L1,
medium->L2/L3 and hard->L4/L5 alternating by index; xhard = L6 and xxhard = L7
live in xhard.py):
  L1: floor 5x5, 4x4 or 3x5 (drawn 2:1:1) and the reference's occupied
      bounding box is the whole floor (cubes in its first and last row and
      in its first and last column; inner rows and columns may be empty),
      height <= 2, 4-8 cubes on 5x5 and exactly 9 on 4x4 and 3x5, yaw 90 or
      270 degrees, three 2-3-cell edit distractors (cube count may change),
      each shown at a random yaw.  (A 3-4 x 4 base with 4-8 cubes has too few
      reference classes for a 25,000-item cell; a 5-long footprint fits no
      L2-L5 grid, and a 4x4 footprint with 9 cubes is no L2-L5 reference, so
      no easy reference can repeat one of another tier.)
  L2: base 2-3 x 2-3, 5-8 cubes, any yaw, one mirror + two 2-cell edits.
  L3: base 3 x 3-4, 7-10 cubes, one mirror + two count-preserving 1-2-cube
      moves.
  L4: base 3 x 3-4, 7-12 cubes, proposal symmetrized left-right with one
      breaking edit, then trimmed (rarely still near-symmetric); one mirror +
      two 1-cube moves.
  L5: base 3-4 x 3-4, 10-14 cubes, height <= 4, one mirror + two 1-cube moves.
  "Any" yaw is 90/180/270.  Cube counts are trimmed/padded into range by
  random one-cube pops/pushes, so most references sit at the upper bound.
  Every mirror of H has one canonical form, so a rung carries exactly one
  mirror distractor (L2-L5; the reference is resampled if the guard refuses
  it).

Each problem is a pure function of (seed, idx): random.Random(f"{seed}:{idx}").
"""
from __future__ import annotations

import itertools
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso
from shared.drawing.fonts import load_bundled_font


def derive_rng(seed, idx):
    return random.Random(f"{seed}:{idx}")


# --------------------------------------------------------------------------- #
# Heightmap algebra
# --------------------------------------------------------------------------- #
def rot(H, k):
    """Rotate heightmap 90deg * k (viewpoint yaw). Returns a list-of-lists."""
    return np.rot90(np.asarray(H), k).tolist()


def mirror(H):
    """Mirror image (left-right flip) -- NOT reachable by any yaw rotation."""
    return np.asarray(H)[:, ::-1].tolist()


def occupied_extent(H):
    """Discard empty border cells: translating a cube structure preserves it."""
    a = np.asarray(H)
    rows, cols = np.where(a > 0)
    if not len(rows):
        return np.zeros((1, 1), dtype=a.dtype)
    return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def canon(H):
    """Canonical occupied structure under translation and the four yaw rotations."""
    a = occupied_extent(H)
    return min((np.rot90(a, k).shape, np.ascontiguousarray(np.rot90(a, k)).tobytes())
               for k in range(4))


def cube_count(H):
    return int(np.asarray(H).sum())


def is_chiral(H):
    return canon(mirror(H)) != canon(H)


def has_rot_symmetry(H):
    A = occupied_extent(H)
    for k in (1, 2, 3):
        R = np.rot90(A, k)
        if R.shape == A.shape and np.array_equal(R, A):
            return True
    return False


# --------------------------------------------------------------------------- #
# Structure sampling per rung
# --------------------------------------------------------------------------- #
RUNGS = {
    # floors (L1 only): ((rows, cols), (min, max cubes), draw weight).
    # Each proposal draws its floor grid and cube bound (sample_heightmap);
    # generate_one keeps it only if its occupied bounding box is the whole
    # floor (cubes in the first and last row and in the first and last
    # column; inner rows/columns may be empty) and its cube count is within
    # the floor's bound.  rows/cols/cubes are only the envelope of the floors.
    1: dict(rows=(3, 5), cols=(4, 5), hmax=2, cubes=(4, 9),  yaws=(1, 3),
            mirror_d=0, edit_cells=(2, 3), preserve_count=False,
            floors=(((5, 5), (4, 8), 2), ((4, 4), (9, 9), 1), ((3, 5), (9, 9), 1))),
    2: dict(rows=(2, 3), cols=(2, 3), hmax=3, cubes=(5, 8),  yaws=(1, 2, 3),
            mirror_d=1, edit_cells=(2, 2), preserve_count=False),
    3: dict(rows=(3, 3), cols=(3, 4), hmax=3, cubes=(7, 10), yaws=(1, 2, 3),
            mirror_d=1, edit_cells=(1, 2), preserve_count=True),
    # mirror_d: mirror-image distractors.  All mirror images of H are yaws of
    # one another (one canonical form), so a second mirror is always a
    # duplicate.
    4: dict(rows=(3, 3), cols=(3, 4), hmax=3, cubes=(7, 12), yaws=(1, 2, 3),
            mirror_d=1, edit_cells=(1, 1), preserve_count=True, near_sym=True),
    5: dict(rows=(3, 4), cols=(3, 4), hmax=4, cubes=(10, 14), yaws=(1, 2, 3),
            mirror_d=1, edit_cells=(1, 1), preserve_count=True),
}


def draw_floor(rng, cfg):
    """L1: one floor grid and its cube bound, drawn by weight."""
    u = rng.randrange(sum(w for _, _, w in cfg["floors"]))
    for grid, cubes, w in cfg["floors"]:
        if u < w:
            return grid, cubes
        u -= w
    raise AssertionError("unreachable")


def floor_cubes(cfg, H):
    """Cube bound of H's floor grid (the union over the floor entries with that grid)."""
    shape = np.asarray(H).shape
    bounds = [c for grid, c, _ in cfg["floors"] if tuple(grid) == shape]
    return min(lo for lo, _ in bounds), max(hi for _, hi in bounds)


def sample_heightmap(rng, cfg):
    if cfg.get("floors"):
        (N, M), (lo, hi) = draw_floor(rng, cfg)
    else:
        N = rng.randint(*cfg["rows"])
        M = rng.randint(*cfg["cols"])
        lo, hi = cfg["cubes"]
    hmax = cfg["hmax"]
    H = [[0] * M for _ in range(N)]
    for j in range(N):
        for i in range(M):
            if rng.random() < 0.7:
                r = rng.random()
                H[j][i] = 1 if r < 0.5 else (2 if r < 0.85 else rng.randint(2, hmax))
    if cfg.get("near_sym"):
        # symmetrize left-right, then break symmetry with a single edit below
        A = np.asarray(H)
        A = np.maximum(A, A[:, ::-1])
        H = A.tolist()
        i = rng.randrange(M)
        j = rng.randrange(N)
        H[j][i] = 0 if H[j][i] > 0 else rng.randint(1, hmax)
    # trim/pad cube count into range by random pops/pushes
    for _ in range(60):
        c = cube_count(H)
        if lo <= c <= hi:
            break
        j = rng.randrange(N); i = rng.randrange(M)
        if c > hi and H[j][i] > 0:
            H[j][i] -= 1
        elif c < lo and H[j][i] < hmax:
            H[j][i] += 1
    return H


def edit_distractor(H, rng, n_cells, preserve_count, hmax):
    """Edit 1-2 column heights; optionally move cubes so the count is kept."""
    A = [row[:] for row in H]
    N, M = len(A), len(A[0])
    if preserve_count:
        for _ in range(n_cells):
            srcs = [(i, j) for j in range(N) for i in range(M) if A[j][i] > 0]
            dsts = [(i, j) for j in range(N) for i in range(M) if A[j][i] < hmax]
            if not srcs or not dsts:
                return None
            si, sj = rng.choice(srcs)
            di, dj = rng.choice([d for d in dsts if d != (si, sj)] or dsts)
            A[sj][si] -= 1
            A[dj][di] += 1
    else:
        cells = [(i, j) for j in range(N) for i in range(M)]
        rng.shuffle(cells)
        for (i, j) in cells[:n_cells]:
            old = A[j][i]
            choices = [c for c in (0, old - 1, old + 1, rng.randint(0, hmax))
                       if 0 <= c <= hmax and c != old]
            if choices:
                A[j][i] = rng.choice(choices)
    if np.asarray(A).sum() == 0 or A == H:
        return None
    return A


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
PALETTES = [(205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160)]


def render_h(H, px, base, margin, floor=True):
    img, _ = iso.render_stack(H, size=px, base_rgb=base, margin=margin,
                              draw_floor=floor, supersample=2)
    return img


def render_check(H, base):
    """Small fixed-size render used for exact pixel-ambiguity checks."""
    return render_h(H, 224, base, 6, floor=False).tobytes()


# --------------------------------------------------------------------------- #
# Reference-side ambiguity guard
# --------------------------------------------------------------------------- #
GUARD_VERSION = "refguard"
GUARD_STATE_LIMIT = 64
GUARD_PAD = 5              # window padding in final px (> the 3 px LANCZOS support)
CONTRAST, AREA, CORE = 32, 24, 6   # extra.difference3d's readable rule (area 24, core 6)
MARGIN_FACE = 0.5          # margins below are at least half a cube top face at the panel's scale
FACE_SHADES = iso.SHADES           # top, left, right
FACE_TOL = 30              # L1 distance (sum over RGB) for a pixel to count as a face shade or white


def option_px(size):
    _, slots = layout_slots(size)
    x0, y0, x1, y1 = slots[0]
    return min(x1 - x0, y1 - y0) - 6


def readable_components(mask):
    """(labels, readable labels) of a contrast mask under difference3d's rule:
    8-connected, area >= AREA, >= CORE pixels left after a 3x3 erosion."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return lab, set()
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    cores = np.bincount(lab[ndimage.binary_erosion(mask, structure=np.ones((3, 3)))], minlength=n + 1)
    return lab, {int(k) for k in np.flatnonzero((areas >= AREA) & (cores >= CORE)) if k}


def readable_px(a, b):
    """Total pixels of the readable difference components between two renders."""
    mask = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=2) >= CONTRAST
    lab, good = readable_components(mask)
    return int(np.isin(lab, sorted(good)).sum()) if good else 0


def face_classes(img, base):
    """Per pixel: 0 top, 1 left, 2 right (palette colour x FACE_SHADES), 3 white
    background, or -1 for anything farther than FACE_TOL (L1) from all four --
    the dark outlines and their anti-aliased fringe."""
    a = np.asarray(img, np.int16)
    refs = [[max(0, min(255, int(c * f))) for c in base] for f in FACE_SHADES] + [[255, 255, 255]]
    d = np.stack([np.abs(a - np.asarray(r, np.int16)).sum(axis=2) for r in refs], axis=-1)
    cls = d.argmin(axis=-1)
    cls[d.min(axis=-1) > FACE_TOL] = -1
    return cls


def face_level_px(a, b, base, a_cls=None):
    """Face-level difference between two renders: pixels that BOTH show a face
    or the background (outline pixels excluded), of different classes, at
    readable contrast (max channel difference >= CONTRAST), and not within one
    pixel (8-neighbourhood) of an outline pixel in BOTH renders.

    The last rule drops the anti-aliased fringe of an outline that sits a pixel
    or two apart in the two renders: that fringe blends into colours within
    FACE_TOL of a face shade, and a whole option panel redrawn 0.5-1 px off
    would otherwise score 0.35-2.6 cube top faces at the delivered sizes.  With
    it the same shifts score at most 0.16 of a face
    (release_checks/face_margin.py), while a region one render shows open
    (white where the other has a face) still counts up to its outline.
    readable_px, by contrast, counts the whole displaced outline."""
    ca = face_classes(a, base) if a_cls is None else a_cls
    cb = face_classes(b, base)
    contrast = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=2) >= CONTRAST
    k = np.ones((3, 3), bool)
    fringe = ndimage.binary_dilation(ca < 0, structure=k) & ndimage.binary_dilation(cb < 0, structure=k)
    return int(((ca >= 0) & (cb >= 0) & (ca != cb) & contrast & ~fringe).sum())


def _meets(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class ReferenceGuard:
    """What the delivered REFERENCE panel leaves open, and whether a
    distractor could be explained by it.

    The option pixel guard only compares options with rotations of H.  A
    column of H that is hidden (or nearly hidden) in the reference can take
    other heights without changing what the viewer sees there, and a distractor
    that is a rotation of such a variant is a second correct answer.  This
    guard renders, at the delivered reference size
    and with H's own camera, every single-column height change 0..max+2 of H;
    the ones that make no readable difference (the difference3d rule) define
    the hidden heights.  Their combinations are the reference-consistent
    variants.  A distractor is refused when it is a rotation of a variant, or
    when its panel (delivered option size, its own camera) is not readably
    different from some 0/90/180/270-degree view of some variant, drawn with
    the same camera both in place and translated so the silhouette corners line
    up, or when it differs from the aligned views of H itself by readable
    regions totalling less than half a cube top face (MARGIN_FACE).

    A face-level margin: against every one of those views (all
    variants, H included, both cameras) D must differ by at least MARGIN_FACE
    top faces of face_level_px -- pixels that both renders show as a face or
    the background, of different classes, at readable contrast, outside the
    fringe the two renders' outlines share (readable_px alone also counts
    outlines displaced by 1-2 px).

    The guard also refuses (variants = None, the caller resamples) references
    whose panel allows a change the single-column search cannot see: two cells
    -- of the grid or of the drawn floor's front pad row/column -- whose joint
    height change is not readably different although one of the two single
    changes is (e.g. a column moved one cell nearer along the view diagonal and
    one cube up: its top lands 6 base px from the old one), or a pad cell that
    can hold cubes unseen.  References with more than GUARD_STATE_LIMIT variants
    are resampled too.  Renders here are windows of the panel drawn with the
    panel's camera; a window reaches GUARD_PAD px beyond every pixel the
    changed columns can draw, so windowed and full-panel comparisons agree.
    """

    def __init__(self, H, size, base):
        self.H, self.base = H, base
        self.ref_px = layout_slots(size)[0]
        self.opt_px = option_px(size)
        self.info = iso.placement(H, self.ref_px, margin=16)
        N, M = len(H), len(H[0])
        self.hcap = max(map(max, H)) + 2
        self._base = {}
        cells = [(j, i) for j in range(N + 1) for i in range(M + 1)]   # grid plus the floor's front pad
        self.single = {}
        grid_domains, pad_hidden = {}, []
        for c in cells:
            h0 = self._h(c)
            vals = [h0]
            for v in range(self.hcap + 1):
                if v == h0:
                    continue
                T = self._grid(((c, v),))
                if cube_count(T) == 0:
                    continue
                win = self._box(c, max(v, h0))
                mask = self._mask(T, win)
                lab, good = readable_components(mask)
                boxes = [(win[0] + sl[1].start, win[1] + sl[0].start, win[0] + sl[1].stop, win[1] + sl[0].stop)
                         for k, sl in enumerate(ndimage.find_objects(lab), 1) if k in good]
                self.single[(c, v)] = (win, mask, boxes)
                if not boxes:
                    vals.append(v)
            if len(vals) > 1:
                if c[0] < N and c[1] < M:
                    grid_domains[c] = vals
                else:
                    pad_hidden.append(("pad", c, vals[1:]))
        self.hidden = {c: d[1:] for c, d in grid_domains.items()}
        self.joint = pad_hidden + self._pairs(cells, grid_domains)
        self.variants = None
        if self.joint:
            return
        n = 1
        for d in grid_domains.values():
            n *= len(d)
        if n > GUARD_STATE_LIMIT:
            return
        self.variants = []
        for combo in itertools.product(*grid_domains.values()):
            S = [row[:] for row in H]
            for (j, i), v in zip(grid_domains, combo):
                S[j][i] = v
            if cube_count(S):
                self.variants.append(S)
        self.canons = {canon(S) for S in self.variants}
        self.views = [rot(S, k) for S in self.variants for k in range(4)]

    # -- windowed renders of the reference panel
    def _h(self, c):
        N, M = len(self.H), len(self.H[0])
        return self.H[c[0]][c[1]] if c[0] < N and c[1] < M else 0

    def _grid(self, changes):
        N, M = len(self.H), len(self.H[0])
        big = any(c[0] >= N or c[1] >= M for c, _ in changes)
        T = [row[:] + ([0] if big else []) for row in self.H] + ([[0] * (M + 1)] if big else [])
        for (j, i), v in changes:
            T[j][i] = v
        return T

    def _box(self, c, h):
        """Final-pixel window holding column c up to height h, padded."""
        info, ss = self.info, self.info["ss"]
        (ox, oy), hw, hh, ch = info["origin"], info["hw"], info["hh"], info["ch"]
        j, i = c
        x0, x1 = (ox + (i - j - 1) * hw) / ss, (ox + (i - j + 1) * hw) / ss
        y0, y1 = (oy + (i + j) * hh - h * ch) / ss, (oy + (i + j + 2) * hh) / ss
        return (max(0, int(np.floor(x0)) - GUARD_PAD), max(0, int(np.floor(y0)) - GUARD_PAD),
                min(self.ref_px, int(np.ceil(x1)) + GUARD_PAD), min(self.ref_px, int(np.ceil(y1)) + GUARD_PAD))

    def _render(self, T, win):
        """Window `win` of the reference panel drawn with H's camera and H's floor."""
        info, ss = self.info, self.info["ss"]
        hw, hh, ch = info["hw"], info["hh"], info["ch"]
        x0, y0, x1, y1 = win
        img = Image.new("RGB", ((x1 - x0) * ss, (y1 - y0) * ss), "white")
        d = ImageDraw.Draw(img)
        origin = (info["origin"][0] - x0 * ss, info["origin"][1] - y0 * ss)
        iso.draw_grid_floor(d, len(self.H), len(self.H[0]), origin, hw, hh, ch, w=ss, pad=1)
        W, Hc = img.size
        for i, j in sorted(((i, j) for j in range(len(T)) for i in range(len(T[0]))), key=sum):
            cx = origin[0] + (i - j) * hw
            if cx + hw < -2 * ss or cx - hw > W + 2 * ss:
                continue                     # the column draws nothing in the window
            for k in range(T[j][i]):
                if origin[1] + (i + j + 2) * hh - k * ch < -2 * ss or origin[1] + (i + j) * hh - (k + 1) * ch > Hc + 2 * ss:
                    continue
                iso.draw_cube(d, i, j, k, origin, base_rgb=self.base, hw=hw, hh=hh, ch=ch, edge_w=ss)
        return np.asarray(img.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)).astype(np.int16)

    def _mask(self, T, win):
        if win not in self._base:
            self._base[win] = self._render(self.H, win)
        return np.abs(self._render(T, win) - self._base[win]).max(axis=2) >= CONTRAST

    def _pairs(self, cells, domains):
        """Pairs of cells whose joint change is unreadable although a single change is readable.
        Outside everything cell b can draw the joint render equals the render with only a
        changed, so a readable component of a's change that survives there cannot be cancelled."""
        found = []
        wmax = {c: self._box(c, self.hcap) for c in cells}
        memo = {}

        def survives(key, other):
            if (key, other) not in memo:
                (x0, y0, x1, y1), mask, boxes = self.single[key]
                ox0, oy0, ox1, oy1 = wmax[other]
                if not boxes:
                    ok = True
                elif not all(_meets(bx, wmax[other]) for bx in boxes):
                    ok = False
                else:
                    m = mask.copy()
                    m[max(0, oy0 - y0):max(0, oy1 - y0), max(0, ox0 - x0):max(0, ox1 - x0)] = False
                    ok = int(m.sum()) < AREA or not readable_components(m)[1]
                memo[(key, other)] = ok
            return memo[(key, other)]

        for x in range(len(cells)):
            for y in range(x + 1, len(cells)):
                a, b = cells[x], cells[y]
                if not _meets(wmax[a], wmax[b]):
                    continue
                va_list = [v for v in range(self.hcap + 1) if (a, v) in self.single and survives((a, v), b)]
                vb_list = [v for v in range(self.hcap + 1) if (b, v) in self.single and survives((b, v), a)]
                for va in va_list:
                    wa = self._box(a, max(va, self._h(a)))
                    for vb in vb_list:
                        if va in domains.get(a, ()) and vb in domains.get(b, ()):
                            continue        # a combination of hidden heights: a variant
                        wb = self._box(b, max(vb, self._h(b)))
                        if not (all(_meets(bx, wb) for bx in self.single[(a, va)][2])
                                and all(_meets(bx, wa) for bx in self.single[(b, vb)][2])):
                            continue
                        T = self._grid(((a, va), (b, vb)))
                        if cube_count(T) == 0:
                            continue
                        win = (min(wa[0], wb[0]), min(wa[1], wb[1]), max(wa[2], wb[2]), max(wa[3], wb[3]))
                        if not readable_components(self._mask(T, win))[1]:
                            found.append(("pair", a, va, b, vb))
        return found

    def accepts(self, D):
        return self.refusal(D) is None

    def refusal(self, D):
        """None if distractor D may be shown, else why not ("variant": a yaw of
        a reference-consistent structure; "view": not readably different from
        a view of one; "margin": readable area < MARGIN_FACE top faces from an
        aligned view of H; "face": face-level difference < MARGIN_FACE top faces
        from some view of some reference-consistent structure)."""
        if canon(D) in self.canons:
            return "variant"
        info = iso.placement(D, self.opt_px, margin=4)
        shown = iso.render_fixed(D, self.opt_px, info, base_rgb=self.base, draw_floor=False)
        shown_cls = face_classes(shown, self.base)
        target = silhouette(D)
        face = 2 * info["hw"] * info["hh"] / info["ss"] ** 2
        nearest = None
        for R in self.views:
            # Same camera as D's panel, both in R's own grid position and
            # translated so R's silhouette corners land on D's (a variant
            # with an emptied hidden column has a smaller footprint).
            for cam in (info, shifted_camera(info, silhouette(R), target)):
                if cam is None:
                    continue
                other = iso.render_fixed(R, self.opt_px, cam, base_rgb=self.base, draw_floor=False)
                if not extra.difference3d(shown, other, area=24, core=6)["readable"]:
                    return "view"
                fl = face_level_px(shown, other, self.base, shown_cls)
                nearest = fl if nearest is None else min(nearest, fl)
        # Margin: more than a readable speck must separate D from
        # every silhouette-aligned view of H (a view whose corners cannot be
        # aligned differs by a whole visible face).
        for k in range(4):
            R = rot(self.H, k)
            have = silhouette(R)
            cam = info if have == target else shifted_camera(info, have, target)
            if cam is None:
                continue
            other = iso.render_fixed(R, self.opt_px, cam, base_rgb=self.base, draw_floor=False)
            if readable_px(shown, other) < MARGIN_FACE * face:
                return "margin"
        # Face-level margin: readable_px also counts outlines displaced by 1-2 px,
        # so a distractor could clear the margin above with a sliver of real face
        # evidence against a view of a hidden-cube variant.  Every view of every
        # reference-consistent
        # structure (H included), with both cameras, must differ from D by at
        # least half a top face of face-level pixels.
        if nearest is not None and nearest < MARGIN_FACE * face:
            return "face"
        return None


def silhouette(H):
    """Front, right and left extremes of the occupied cells: max(i+j),
    max(i-j), max(j-i).  These columns are never covered in the view."""
    cells = np.argwhere(np.asarray(H) > 0)
    j, i = cells[:, 0], cells[:, 1]
    return int((i + j).max()), int((i - j).max()), int((j - i).max())


def shifted_camera(info, have, want):
    """info with its origin moved by the lattice translation (di, dj) that maps
    silhouette `have` onto `want`; None if no translation does or it is zero."""
    s, d = want[0] - have[0], want[1] - have[1]
    if (s + d) % 2 or want[2] - have[2] != -d:
        return None
    di, dj = (s + d) // 2, (s - d) // 2
    if di == 0 and dj == 0:
        return None
    ox, oy = info["origin"]
    return {**info, "origin": (ox + (di - dj) * info["hw"], oy + (di + dj) * info["hh"])}


def layout_slots(size):
    """Reference zone on top, 2x2 option slots below. Returns (ref_px, slots)."""
    ref_px = int(size * 0.40)
    strip_top = ref_px + 4
    slot_w = (size - 12) // 2
    slot_h = (size - strip_top - 10) // 2
    slots = []
    for k in range(4):
        r, c = divmod(k, 2)
        x0 = 6 + c * slot_w + (2 if c else 0)
        y0 = strip_top + 4 + r * slot_h + (2 if r else 0)
        slots.append((x0, y0, x0 + slot_w - 4, y0 + slot_h - 4))
    return ref_px, slots


def compose(H_ref, options, size, base):
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    ref_px, slots = layout_slots(size)
    ref = render_h(H_ref, ref_px, base, 16)
    canvas.paste(ref, ((size - ref_px) // 2, 2))
    d.line([(8, ref_px + 2), (size - 8, ref_px + 2)], fill=(230, 230, 234),
           width=2)

    f = load_bundled_font(18, "sans-bold")
    for k, opt in enumerate(options):
        x0, y0, x1, y1 = slots[k]
        panel_px = min(x1 - x0, y1 - y0) - 6
        panel = render_h(opt, panel_px, base, 4, floor=False)
        canvas.paste(panel, (x0 + (x1 - x0 - panel_px) // 2,
                             y0 + (y1 - y0 - panel_px) // 2))
        d.rounded_rectangle([x0, y0, x1, y1], radius=8,
                            outline=(225, 225, 230), width=2)
        d.text((x0 + 7, y0 + 4), "ABCD"[k], fill=(30, 30, 30), font=f)
    return canvas


# --------------------------------------------------------------------------- #
# Problem assembly
# --------------------------------------------------------------------------- #
def build_options(H, rng, cfg, guard=None):
    """Return (option_heightmaps, correct_index, yaw_k, kinds) or None."""
    base_canon = canon(H)
    mirror_canon = canon(mirror(H))
    yaw_k = rng.choice(cfg["yaws"])
    truth = rot(H, yaw_k)

    truth_renders = {render_check(rot(H, k), PALETTES[0]) for k in range(4)}
    seen_canon = {base_canon}
    seen_px = set(truth_renders)
    distractors = []  # (heightmap, kind)

    def try_add(D, kind):
        if D is None:
            return False
        c = canon(D)
        if c in seen_canon:
            return False
        px = render_check(D, PALETTES[0])
        if px in seen_px:
            return False
        if guard is not None and not guard.accepts(D):
            return False
        seen_canon.add(c)
        seen_px.add(px)
        # label by structure: an edit can coincide with the mirror image
        distractors.append((D, "mirror" if c == mirror_canon else kind))
        return True

    # mirror-based distractors (of a random rotation, then maybe rotated again)
    for _ in range(cfg["mirror_d"] * 4):
        if sum(1 for _, k in distractors if k == "mirror") >= cfg["mirror_d"]:
            break
        try_add(rot(mirror(rot(H, rng.randrange(4))), rng.randrange(4)), "mirror")

    # edit distractors
    attempts = 0
    while len(distractors) < 3 and attempts < 120:
        attempts += 1
        n = rng.randint(*cfg["edit_cells"])
        D = edit_distractor(H, rng, n, cfg["preserve_count"], cfg["hmax"])
        if D is not None:
            try_add(rot(D, rng.randrange(4)), "edit")

    if len(distractors) < 3:
        return None
    if cfg["mirror_d"] and sum(k == "mirror" for _, k in distractors[:3]) != cfg["mirror_d"]:
        return None   # the guard refused the mirror: resample rather than ship 0 mirrors
    opts = [(truth, "truth")] + distractors[:3]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [opts[o] for o in order]
    correct_index = order.index(0)
    return ([h for h, _ in shuffled], correct_index, yaw_k,
            [k for _, k in shuffled])


def difficulty_for(rung):
    return {1: "easy", 2: "medium", 3: "medium", 4: "hard", 5: "hard"}[rung]


QUESTION = (
    'Which option A-D shows the same cube structure as the top panel, with the viewpoint turned '
    'about the vertical axis by 90, 180 or 270 degrees? Some cubes may be hidden; options are '
    'scaled to fit and omit the floor grid. Answer with the letter.'
)


def generate_one(seed, idx, size, difficulty=None):
    """One L1-L5 item: (record, image)."""
    rng = derive_rng(seed, idx)
    rung = extra.base_rung(idx, difficulty)
    cfg = RUNGS[rung]
    # The palette is drawn first so the reference guard measures differences
    # in the colours the viewer will actually see.
    base = PALETTES[rng.randrange(len(PALETTES))]

    for _attempt in range(200):
        H = sample_heightmap(rng, cfg)
        if cfg.get("floors"):
            lo, hi = floor_cubes(cfg, H)
            if not lo <= cube_count(H) <= hi or occupied_extent(H).shape != np.asarray(H).shape:
                continue   # occupied bounding box == the floor; trimming can (rarely) run out of pops
        elif cube_count(H) < cfg["cubes"][0]:
            continue
        if not is_chiral(H) or has_rot_symmetry(H):
            continue
        guard = ReferenceGuard(H, size, base)
        if guard.variants is None:
            continue
        built = build_options(H, rng, cfg, guard)
        if built is None:
            continue
        options, correct_index, yaw_k, kinds = built
        break
    else:
        raise RuntimeError(f"could not build a valid problem for idx={idx}")

    img = compose(H, options, size, base)

    letter = "ABCD"[correct_index]
    cot = (f"Rotating the viewpoint about the vertical axis by {90 * yaw_k} "
           f"degrees turns the reference structure into option {letter}. The "
           f"other options are mirror images or have columns of different "
           f"heights, so no rotation can match them.")
    rec = dict(
        id=f"viewpoint_rotation_{idx:06d}",
        task_type="spatial_perception",
        subtype="viewpoint_rotation",
        level="high",
        difficulty=difficulty_for(rung),
        question=QUESTION,
        ans_type="choice",
        options=["A", "B", "C", "D"],
        answer=letter,
        cot=cot,
        meta=dict(
            generator_version=extra.VERSION,
            rung=f"L{rung}",
            heightmap=H,
            cube_count=cube_count(H),
            yaw_degrees=90 * yaw_k,
            option_heightmaps=options,
            option_kinds=kinds,
            correct_index=correct_index,
            resolution=size,
            palette_rgb=list(base),
            reference_size=guard.ref_px,
            option_size=guard.opt_px,
            reference_guard=GUARD_VERSION,
            reference_hidden_cells=len(guard.hidden),
            reference_variants=len(guard.variants),
        ),
    )
    return rec, img
