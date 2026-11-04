"""Independent verifier for viewpoint_rotation.

Nothing here imports the generator (generator/generate.py,
shared/drawing/isometric.py or extra_difficulty.py).  The verifier carries its own copy of the
documented conventions -- the 2:1 projection screen x = ox + (i - j) * hw,
y = oy + (i + j) * hh - k * ch with painter's order by i + j, the face shades
(top 1.0, left 0.74, right 0.88 of the palette colour, 20-grey outlines), the
panel layout and the auto-fit placement rule -- and two renderers of its own:
a face-LABEL rasteriser and an RGB painter.  From those it

1. checks the question text (which must show no example answer letter), the
   answer format and the metadata;
2. decodes the image against the metadata: (a) every pixel two or more pixels
   inside a face or the background of the reference panel and of each option
   panel must carry that face's shade, (b) the verifier's own RGB re-render of
   each panel must agree with the delivered pixels, so the picture shows
   exactly the structures in meta, at the delivered resolution, and (c) every
   pixel OUTSIDE the five panels must match the verifier's own drawing of the
   page furniture (white page, separator line, slot outlines and the letters
   A B / C D in the repository's DejaVuSans-Bold), so slot k carries the letter
   'ABCD'[k] and nothing else is drawn on the page;
3. recomputes the gold letter as the only option whose structure is a
   90/180/270-degree yaw of the reference (own canonical form up to
   translation and yaw), never reading the stored letter;
4. proves uniqueness at the delivered resolution.  It finds every column of
   the reference whose height can change (to anything in 0..max+2) without a
   readable change of the REFERENCE panel -- the structures the reference
   cannot rule out are the combinations of those heights -- and shows that
   each distractor panel differs readably from every 0/90/180/270-degree view
   of every such structure, at every translation that lines up the
   silhouettes.  A distractor that only differs from a yawed truth in places
   the viewer cannot see, or that IS a yaw of a structure the reference allows
   (a hidden column filled in or emptied), is reported.  The search covers
   every cell of the heightmap grid; the question states that cubes stand on
   the drawn floor grid, whose extra front row and column (the pad) can never
   hide a cube (nothing is drawn in front of their outer faces), so no other
   cell can hold a hidden cube.  Heights above max(H) cannot hide either (the
   top face would be above every occluder); max+1 and max+2 are tested anyway.
   Changes that are readable one at a time can cancel each other (a column
   one cell nearer along the view diagonal and one cube taller has its top
   6 base px from the old one, hh:ch = 17:40; top and right faces differ by
   less than the 32-level contrast), so every pair of cells -- the grid and
   the floor's front pad row/column -- is searched too, and a reference that
   allows a jointly unreadable two-cell change is refused;
5. requires each distractor to differ from the nearest silhouette-aligned view
   of the reference structure by readable regions totalling at least half a
   cube top face at that option panel's scale (a margin, not only a readable
   speck), and -- because readable regions include outlines that merely moved
   by a pixel or two -- from every 0/90/180/270-degree view of EVERY structure
   the reference allows (H included), in place and silhouette-aligned, by at
   least half a cube top face of FACE-LEVEL difference: pixels that both panels
   show as a face or the background (each within L1 30 of a shade or white),
   of different classes, at contrast >= 32, and not within one pixel of an
   outline in both panels (face_margin);
6. checks the tier rules the record's tier states (rung, grid, cube count,
   height cap, yaw set, and the distractor recipe: number of mirror images and
   the kind of column edit), from the structures alone.

"Readable difference" (the repository's difference3d rule): the per-pixel
maximum channel difference is >= 32 on an 8-connected region of at least 24
pixels that keeps at least 6 pixels after a 3x3 erosion.  Renders that are
compared share one camera (the reference camera for reference variants, the
distractor's own camera for option panels), so auto-fit rescaling cannot
create or hide a difference.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import functools
import itertools
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CORRECTNESS = (
    "Decodes all five panels against meta at the delivered resolution (deep-interior face shades from an own label "
    "rasteriser, plus an own RGB re-render), checks every pixel outside the panels against an own drawing of the page "
    "furniture (slot k must carry the letter ABCD[k]), then recomputes the gold letter as the only option whose "
    "structure is a 90/180/270-degree yaw of the reference heightmap (own canonical form up to translation and yaw)."
)
UNIQUENESS = (
    "Finds every reference column whose height (0..max+2) can change without a readable change of the reference "
    "panel, refuses references with an unreadable one-cell diagonal relocation (depth/height alias), and proves each "
    "distractor panel differs readably (difference3d: >=32 levels on a >=24 px region with a >=6 px core, same "
    "camera, silhouettes aligned) from every 0/90/180/270-degree view of every structure those hidden heights allow, "
    "from the aligned views of the reference by at least half a cube top face of readable area, and from every view "
    "of every allowed structure (in place and aligned) by at least half a cube top face of face-level difference "
    "(outline pixels and their shared fringe excluded); the reference must be chiral and without rotational "
    "self-symmetry."
)

QUESTION = (
    'Which option A-D shows the same cube structure as the top panel, with the viewpoint turned '
    'about the vertical axis by 90, 180 or 270 degrees? Some cubes may be hidden; options are '
    'scaled to fit and omit the floor grid. Answer with the letter.'
)
LETTERS = ["A", "B", "C", "D"]
CHOICE_RANGE = "A-D"     # names the whole choice set; it is not an example answer

SS = 2                     # the delivered renderer draws at 2x and downsamples (Lanczos)
Q = 4                      # label subsamples per final pixel, per axis
BASE = (34, 17, 40)        # half-tile width, half-tile height, cube height at 1x
SHADE = (1.0, 0.74, 0.88)  # top, left (+j side), right (+i side)
EDGE = (20, 20, 20)
FLOOR_RGB = (200, 205, 212)
CONTRAST, AREA, CORE = 32, 24, 6
REF_MARGIN, OPT_MARGIN = 16, 4
HCAP_EXTRA = 2
PAD = 5                    # window padding (> Lanczos support of 3 px)
COMBO_LIMIT = 4096
OPTIONAL_LIMIT = 12        # hidden columns that may be emptied; more -> refuse rather than truncate the search
FIDELITY_TOL = 10          # deep-interior shade tolerance
RERENDER_TOL = 6           # own RGB re-render vs delivered pixels
FLOOR_LABEL = 1 << 30
MARGIN_FACE = 0.5          # margins (readable px vs aligned views of H; face-level px vs every allowed view) in top faces
FACE_TOL = 30              # L1 distance (sum over RGB) for a pixel to count as a face shade or the white background
# Page furniture outside the panels (documented layout): separator, slot outlines, option letters.
PAGE_RGB, SEP_RGB, SLOT_RGB, LETTER_RGB = (255, 255, 255), (230, 230, 234), (225, 225, 230), (30, 30, 30)
LETTER_PX = 18
LETTER_FONT = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')
CHROME_TOL = 6


# --------------------------------------------------------------------------- geometry
def layout(size):
    """(ref box, option boxes, option slots); a box is (x, y, px)."""
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
    ref = ((size - ref_px) // 2, 2, ref_px)
    opts = []
    for x0, y0, x1, y1 in slots:
        p = min(x1 - x0, y1 - y0) - 6
        opts.append((x0 + (x1 - x0 - p) // 2, y0 + (y1 - y0 - p) // 2, p))
    return ref, opts, slots


def _bbox(H, hw, hh, ch):
    N, M = len(H), len(H[0])
    xs, ys = [], []
    for i, j in ((0, 0), (M + 1, 0), (0, N + 1), (M + 1, N + 1)):   # floor footprint, padded by one cell
        xs.append((i - j) * hw)
        ys.append((i + j) * hh)
    for j in range(N):
        for i in range(M):
            h = H[j][i]
            if h <= 0:
                continue
            for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for k in (h, 0):
                    xs.append((i + di - j - dj) * hw)
                    ys.append((i + di + j + dj) * hh - k * ch)
    return min(xs), min(ys), max(xs), max(ys)


def camera(H, px, margin):
    """Auto-fit placement of one panel (canvas units = SS x final pixels)."""
    hw, hh, ch = (v * SS for v in BASE)
    x0, y0, x1, y1 = _bbox(H, hw, hh, ch)
    canvas = px * SS
    scale = min(1.0, (canvas - 2 * margin * SS) / max(x1 - x0, y1 - y0, 1))
    hw, hh, ch = max(2, int(hw * scale)), max(1, int(hh * scale)), max(2, int(ch * scale))
    x0, y0, x1, y1 = _bbox(H, hw, hh, ch)
    return dict(ox=(canvas - (x1 - x0)) / 2 - x0, oy=(canvas - (y1 - y0)) / 2 - y0, hw=hw, hh=hh, ch=ch, px=px)


def _xy(cam, i, j, k):
    """Canvas (SS) coordinates of lattice point (i, j, level k)."""
    return cam["ox"] + (i - j) * cam["hw"], cam["oy"] + (i + j) * cam["hh"] - k * cam["ch"]


def column_window(cam, i, j, h):
    """Final-pixel box [x0, x1) x [y0, y1) that holds column (i, j) up to height h, padded."""
    x0 = (cam["ox"] + (i - j - 1) * cam["hw"]) / SS
    x1 = (cam["ox"] + (i - j + 1) * cam["hw"]) / SS
    y0 = (cam["oy"] + (i + j) * cam["hh"] - h * cam["ch"]) / SS
    y1 = (cam["oy"] + (i + j + 2) * cam["hh"]) / SS
    px = cam["px"]
    return (max(0, int(np.floor(x0)) - PAD), max(0, int(np.floor(y0)) - PAD),
            min(px, int(np.ceil(x1)) + PAD), min(px, int(np.ceil(y1)) + PAD))


def _meets(a, b):
    """Boxes (x0, y0, x1, y1) overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _union(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _faces(S, offset):
    """Cubes in painter's order (ascending i + j, then level) with their three faces."""
    di, dj = offset
    N, M = len(S), len(S[0])
    for i, j in sorted(((i, j) for j in range(N) for i in range(M)), key=lambda c: c[0] + c[1]):
        I, J = i + di, j + dj
        h = int(S[j][i])
        for k in range(h):
            top = [(I, J, k + 1), (I + 1, J, k + 1), (I + 1, J + 1, k + 1), (I, J + 1, k + 1)]
            left = [(I, J + 1, k + 1), (I + 1, J + 1, k + 1), (I + 1, J + 1, k), (I, J + 1, k)]
            right = [(I + 1, J, k + 1), (I + 1, J + 1, k + 1), (I + 1, J + 1, k), (I + 1, J, k)]
            yield k == h - 1, (top, left, right)


def render_rgb(S, cam, rgb, offset=(0, 0), floor=False, window=None, grid=None):
    """The verifier's own painter: int16 RGB of `window` (final px) of one panel.
    `grid` = (N, M) draws the floor of an N x M heightmap under a larger S."""
    px = cam["px"]
    wx0, wy0, wx1, wy1 = window or (0, 0, px, px)
    img = Image.new("RGB", ((wx1 - wx0) * SS, (wy1 - wy0) * SS), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def P(p):
        x, y = _xy(cam, *p)
        return (x - wx0 * SS, y - wy0 * SS)

    N, M = grid or (len(S), len(S[0]))
    if floor:
        for a in range(M + 2):
            d.line([P((a, 0, 0)), P((a, N + 1, 0))], fill=FLOOR_RGB, width=SS)
        for b in range(N + 2):
            d.line([P((0, b, 0)), P((M + 1, b, 0))], fill=FLOOR_RGB, width=SS)
    shades = [tuple(max(0, min(255, int(c * f))) for c in rgb) for f in SHADE]
    W, Hh = (wx1 - wx0) * SS, (wy1 - wy0) * SS
    for _, polys in _faces(S, offset):
        pts = [[P(p) for p in poly] for poly in polys]
        xs = [x for poly in pts for x, _ in poly]
        ys = [y for poly in pts for _, y in poly]
        if max(xs) < -2 * SS or min(xs) > W + 2 * SS or max(ys) < -2 * SS or min(ys) > Hh + 2 * SS:
            continue   # the cube draws nothing inside the window
        for poly, col in zip(pts, shades):
            d.polygon(poly, fill=col, outline=EDGE)
        for poly in pts:
            d.line(poly + [poly[0]], fill=EDGE, width=SS)
    img = img.resize((wx1 - wx0, wy1 - wy0), Image.LANCZOS)
    return np.asarray(img).astype(np.int16)


def label_render(S, cam, floor=False):
    """Face labels at Q x Q subsamples per final pixel: 0 background, FLOOR_LABEL floor line,
    1 + 3 * cube + face (0 top, 1 left, 2 right)."""
    px = cam["px"]
    f = Q / SS
    img = Image.new("I", (px * Q, px * Q), 0)
    d = ImageDraw.Draw(img)

    def P(p):
        x, y = _xy(cam, *p)
        return (x * f, y * f)

    N, M = len(S), len(S[0])
    if floor:
        for a in range(M + 2):
            d.line([P((a, 0, 0)), P((a, N + 1, 0))], fill=FLOOR_LABEL, width=Q)
        for b in range(N + 2):
            d.line([P((0, b, 0)), P((M + 1, b, 0))], fill=FLOOR_LABEL, width=Q)
    for n, (is_top, polys) in enumerate(_faces(S, (0, 0))):
        for face, poly in enumerate(polys):
            if face == 0 and not is_top:
                continue  # covered by the cube above
            d.polygon([P(p) for p in poly], fill=1 + 3 * n + face)
    return np.asarray(img)


def face_classes(L):
    """0 top, 1 left, 2 right, 3 background per final pixel; -1 within one pixel of any boundary."""
    h, w = L.shape[0] // Q, L.shape[1] // Q
    sub = L.reshape(h, Q, w, Q)
    lo, hi = sub.min(axis=(1, 3)), sub.max(axis=(1, 3))
    ids = np.where(lo == hi, lo, -1)
    pad = np.pad(ids, 1, constant_values=-2)
    inner = (ids >= 0) & (ids != FLOOR_LABEL)
    for dy in range(3):
        for dx in range(3):
            inner &= pad[dy:dy + h, dx:dx + w] == ids
    return np.where(inner, np.where(ids == 0, 3, (ids - 1) % 3), -1)


def readable_mask(a, b):
    """Pixels of readable difference components between two RGB renders (difference3d rule)."""
    return readable_mask_of(np.abs(a - b).max(axis=2) >= CONTRAST)


def _good_components(mask, lab, n):
    """Labels of the readable components (area >= AREA, core >= CORE after a 3x3 erosion)."""
    if not n:
        return set()
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    cores = np.bincount(lab[ndimage.binary_erosion(mask, structure=np.ones((3, 3)))], minlength=n + 1)
    return {int(k) for k in np.flatnonzero((areas >= AREA) & (cores >= CORE)) if k}


def readable_mask_of(mask):
    """Pixels of readable components of a contrast mask (the difference3d rule)."""
    if mask.sum() < AREA:
        return np.zeros(mask.shape, bool)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return np.isin(lab, sorted(_good_components(mask, lab, n)))


def readable(a, b):
    return bool(readable_mask(a, b).any())


# --------------------------------------------------------------------------- structure algebra
def extent(a):
    a = np.asarray(a)
    nz = np.argwhere(a > 0)
    if not len(nz):
        return np.zeros((1, 1), int)
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
    return a[r0:r1 + 1, c0:c1 + 1]


def canon(a):
    e = extent(a)
    return min((np.rot90(e, k).shape, tuple(int(v) for v in np.rot90(e, k).ravel())) for k in range(4))


def yaw_of(H, O):
    """k in 0..3 with O == rot90(H, k) up to translation, else None."""
    e = extent(O)
    for k in range(4):
        r = extent(np.rot90(H, k))
        if r.shape == e.shape and np.array_equal(r, e):
            return k
    return None


# --------------------------------------------------------------------------- image decode
def _decode_panel(arr, box, S, margin, floor, rgb, name):
    x, y, px = box
    cam = camera(S, px, margin)
    crop = arr[y:y + px, x:x + px].astype(np.int16)
    probs = []
    cls = face_classes(label_render(S, cam, floor=floor))
    deep = ndimage.binary_erosion(cls >= 0, structure=np.ones((3, 3)))
    expect = np.full(cls.shape + (3,), 255, np.int16)
    for c in range(3):
        expect[cls == c] = [max(0, min(255, int(v * SHADE[c]))) for v in rgb]
    err = np.abs(crop - expect).max(axis=2)
    if deep.sum() < 0.5 * px * px:
        probs.append(f"{name}: only {int(deep.sum())} decodable pixels")
    bad = deep & (err > FIDELITY_TOL)
    if bad.any():
        probs.append(f"{name}: {int(bad.sum())} face pixels disagree with the meta structure")
    for c in range(3):
        if not (deep & (cls == c)).any():
            probs.append(f"{name}: no visible {('top', 'left', 'right')[c]} face")
    rer = np.abs(crop - render_rgb(S, cam, rgb, floor=floor)).max(axis=2)
    if (rer > RERENDER_TOL).any():
        probs.append(f"{name}: {int((rer > RERENDER_TOL).sum())} pixels differ from the re-render of meta "
                     f"(max {int(rer.max())})")
    return probs


# --------------------------------------------------------------------------- page furniture
@functools.lru_cache(maxsize=None)
def _letter_font():
    return ImageFont.truetype(str(LETTER_FONT), LETTER_PX)


@functools.lru_cache(maxsize=None)
def chrome(size, letters="ABCD"):
    """The page outside the panels, drawn the documented way: white page, a 2 px
    separator under the reference zone, four rounded slot outlines and the
    letter letters[k] at (x0 + 7, y0 + 4) of slot k in DejaVuSans-Bold 18."""
    ref, _, slots = layout(size)
    img = Image.new("RGB", (size, size), PAGE_RGB)
    d = ImageDraw.Draw(img)
    d.line([(8, ref[2] + 2), (size - 8, ref[2] + 2)], fill=SEP_RGB, width=2)
    f = _letter_font()
    for k, (x0, y0, x1, y1) in enumerate(slots):
        d.rounded_rectangle([x0, y0, x1, y1], radius=8, outline=SLOT_RGB, width=2)
        d.text((x0 + 7, y0 + 4), letters[k], fill=LETTER_RGB, font=f)
    out = np.asarray(img).astype(np.int16)
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=None)
def outside_panels(size):
    ref, opts, _ = layout(size)
    mask = np.ones((size, size), bool)
    for x, y, px in [ref] + opts:
        mask[y:y + px, x:x + px] = False
    mask.setflags(write=False)
    return mask


def label_box(slot):
    x0, y0, _, _ = slot
    return x0 + 3, y0 + 3, x0 + 26, y0 + 28


def read_label(arr, size, k):
    """(letter, mean abs error) of the A-D template closest to slot k's label area."""
    _, _, slots = layout(size)
    a, b, c, d = label_box(slots[k])
    crop = arr[b:d, a:c].astype(np.int16)
    errs = {L: float(np.abs(crop - chrome(size, L * 4)[b:d, a:c]).mean()) for L in LETTERS}
    best = min(errs, key=errs.get)
    return best, errs[best]


def chrome_problems(arr, size):
    """Every pixel outside the five panels must be the documented page furniture
    (within CHROME_TOL levels); slot k must read 'ABCD'[k]."""
    try:
        expect = chrome(size)
    except OSError as exc:
        return [f"environment: cannot load the option-letter font {LETTER_FONT} ({exc})"]
    probs = []
    _, _, slots = layout(size)
    bad = (np.abs(arr.astype(np.int16) - expect).max(axis=2) > CHROME_TOL) & outside_panels(size)
    for k, slot in enumerate(slots):
        a, b, c, d = label_box(slot)
        if bad[b:d, a:c].any():
            got, err = read_label(arr, size, k)
            probs.append(f"option label {LETTERS[k]}: slot {k} reads {got!r} (mean error {err:.1f}), "
                         f"{int(bad[b:d, a:c].sum())} pixels off the expected glyph")
            bad[b:d, a:c] = False
    if bad.any():
        ys, xs = np.nonzero(bad)
        probs.append(f"{int(bad.sum())} pixels outside the panels differ from the page furniture "
                     f"(first at x={int(xs[0])}, y={int(ys[0])})")
    return probs


def format_example_letters(question):
    """Letters the question shows as an example answer: every standalone A-D
    token outside the choice-range marker "(A-D)"."""
    return re.findall(r"(?<![A-Za-z])[A-D](?![A-Za-z])", question.replace(CHOICE_RANGE, " "))


# --------------------------------------------------------------------------- what the reference pins down
class Reference:
    """What the reference panel pins down.

    domain[(j, i)] lists every height 0..max+2 the cell can take, by itself,
    without a readable change of the reference panel (same camera, floor
    drawn).  The structures the reference allows are the combinations of these
    domains; every combination the option-side search relies on is re-tested
    against the whole reference.

    Changes that are readable one at a time can still cancel each other: in
    the 34:17:40 projection a column one cell nearer along the view diagonal
    and one cube taller has the same screen x and its top lands only 6 base px
    from the old one (2*hh - ch), and an exposed cube can stand in for a removed
    one beside it.  `joint` lists every pair of cells -- grid cells and the
    drawn grid's front pad row/column -- and heights 0..max+2 whose joint change
    is not readably different from H although at least one of the two single
    changes is (plus any unreadable single change of a pad cell).  The verifier
    refuses such references, so the allowed structures are exactly the
    combinations of the single-column domains, up to changes of three or more
    columns that are only jointly invisible.

    Pruning is exact: outside the padded window of everything cell b can draw,
    the joint render equals the render with only cell a changed, so if a's
    difference restricted to those pixels already holds a readable component,
    no change of b can cancel it (the component can only grow)."""

    def __init__(self, H, ref_px, rgb):
        self.H, self.rgb = H, rgb
        N, M = H.shape
        self.cam = camera(H.tolist(), ref_px, REF_MARGIN)
        self.hcap = int(H.max()) + HCAP_EXTRA
        self._base = {}
        self.domain = {}
        self.cells = [(j, i) for j in range(N + 1) for i in range(M + 1)]   # grid plus front pad
        self.win = {c: column_window(self.cam, c[1], c[0], self.hcap) for c in self.cells}
        self.single = {}            # (cell, v) -> (window, contrast mask in it, boxes of readable components)
        for c in self.cells:
            h0 = self._h(c)
            vals = [h0]
            for v in range(self.hcap + 1):
                if v == h0:
                    continue
                T = self._grid(((c, v),))
                if not T.sum():
                    continue
                win = column_window(self.cam, c[1], c[0], max(v, h0))
                mask = self._mask(T, win)
                lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
                good = _good_components(mask, lab, n)
                # components that are readable on their own, as (x0, y0, x1, y1) boxes in panel pixels
                boxes = [(win[0] + sl[1].start, win[1] + sl[0].start, win[0] + sl[1].stop, win[1] + sl[0].stop)
                         for k, sl in enumerate(ndimage.find_objects(lab), 1) if k in good]
                self.single[(c, v)] = (win, mask, boxes)
                if not boxes:
                    vals.append(v)
            if len(vals) > 1:
                if c[0] < N and c[1] < M:
                    self.domain[c] = vals
        self.joint = [f"pad cell {c} can hold {vals[1:]} cube(s) unseen"
                      for c, vals in self._pad_domains().items()]
        self.joint += self._pairs()

    def _h(self, c):
        N, M = self.H.shape
        return int(self.H[c]) if c[0] < N and c[1] < M else 0

    def _grid(self, changes):
        N, M = self.H.shape
        big = any(c[0] >= N or c[1] >= M for c, _ in changes)
        T = np.zeros((N + 1, M + 1) if big else (N, M), int)
        T[:N, :M] = self.H
        for c, v in changes:
            T[c] = v
        return T

    def _pad_domains(self):
        N, M = self.H.shape
        out = {}
        for (c, v), (_, _, boxes) in self.single.items():
            if (c[0] == N or c[1] == M) and not boxes:
                out.setdefault(c, [0]).append(v)
        return out

    def _mask(self, T, win):
        """Bool mask (window coordinates) of contrast >= CONTRAST between T and H inside `win`."""
        if win not in self._base:
            self._base[win] = render_rgb(self.H.tolist(), self.cam, self.rgb, floor=True, window=win)
        got = render_rgb(T.tolist(), self.cam, self.rgb, floor=True, window=win, grid=self.H.shape)
        return np.abs(got - self._base[win]).max(axis=2) >= CONTRAST

    def _pairs(self):
        found = []
        cells = self.cells
        blocked = {}

        def survives(key, other):
            """False if (cell, v)'s difference keeps a readable component outside other's window."""
            k = (key, other)
            if k not in blocked:
                (x0, y0, x1, y1), mask, boxes = self.single[key]
                ox0, oy0, ox1, oy1 = self.win[other]
                if not boxes:
                    blocked[k] = True
                elif not all(_meets(b, self.win[other]) for b in boxes):
                    blocked[k] = False       # a readable component lies wholly outside: it stays
                else:
                    m = mask.copy()
                    m[max(0, oy0 - y0):max(0, oy1 - y0), max(0, ox0 - x0):max(0, ox1 - x0)] = False
                    blocked[k] = int(m.sum()) < AREA or not readable_mask_of(m).any()
            return blocked[k]

        for x in range(len(cells)):
            for y in range(x + 1, len(cells)):
                a, b = cells[x], cells[y]
                if not _meets(self.win[a], self.win[b]):
                    continue
                va_list = [v for v in range(self.hcap + 1) if (a, v) in self.single and survives((a, v), b)]
                vb_list = [v for v in range(self.hcap + 1) if (b, v) in self.single and survives((b, v), a)]
                for va in va_list:
                    wa_v = column_window(self.cam, a[1], a[0], max(va, self._h(a)))
                    for vb in vb_list:
                        if va in self.domain.get(a, ()) and vb in self.domain.get(b, ()):
                            continue    # a combination of hidden heights: searched option-side
                        wb_v = column_window(self.cam, b[1], b[0], max(vb, self._h(b)))
                        # each readable component of one change must lie where the other change draws
                        if not (all(_meets(box, wb_v) for box in self.single[(a, va)][2])
                                and all(_meets(box, wa_v) for box in self.single[(b, vb)][2])):
                            continue
                        T = self._grid(((a, va), (b, vb)))
                        if not T.sum():
                            continue
                        win = _union(wa_v, wb_v)
                        if not readable_mask_of(self._mask(T, win)).any():
                            found.append(f"cells {a}->{va} and {b}->{vb} change together without a readable "
                                         f"difference")
        return found

    def consistent(self, T):
        """T (H's grid) is not readably different from H in the reference panel."""
        win = None
        for j, i in np.argwhere(T != self.H):
            win = _union(win, column_window(self.cam, int(i), int(j), max(int(T[j, i]), int(self.H[j, i]))))
        if win is None:
            return True
        return not readable_mask_of(self._mask(T, win)).any()


# --------------------------------------------------------------------------- option-side comparison
def _stats(cells):
    """Silhouette corners of an occupied cell set (row j, col i): front max(i+j), right max(i-j), left max(j-i)."""
    return (max(i + j for j, i in cells), max(i - j for j, i in cells), max(j - i for j, i in cells))


def _rot_where(shape, k):
    """original (j, i) -> (row, col) in np.rot90(array, k)."""
    idx = np.rot90(np.arange(shape[0] * shape[1]).reshape(shape), k)
    return {divmod(int(idx[r, c]), shape[1]): (r, c) for r in range(idx.shape[0]) for c in range(idx.shape[1])}


def _alignments(ref, k, D):
    """Translations (di, dj) that put the front, right and left silhouette corners
    of some reference-allowed structure, yawed by k, on those of D.  A structure
    whose corners do not land on D's differs from D by a whole visible cube.
    None when more than OPTIONAL_LIMIT hidden columns could be emptied (the
    subset search would be truncated, so the caller refuses the item)."""
    H = ref.H
    where = _rot_where(H.shape, k)
    fixed = [where[(int(j), int(i))] for j, i in np.argwhere(H > 0)
             if min(ref.domain.get((int(j), int(i)), [1])) > 0]
    optional = [where[c] for c, vals in ref.domain.items() if min(vals) == 0 and max(vals) > 0]
    if len(optional) > OPTIONAL_LIMIT:
        return None
    target = _stats([tuple(c) for c in np.argwhere(D > 0)])
    out = {}
    for mask in range(1 << len(optional)):
        occ = fixed + [c for b, c in enumerate(optional) if mask >> b & 1]
        if not occ:
            continue
        a, b, c = _stats(occ)
        s, d = target[0] - a, target[1] - b
        if (s + d) % 2 or target[2] - c != -d:
            continue
        out[((s + d) // 2, (s - d) // 2)] = None
    return list(out)


def distractor_explained(ref, D, opt_px):
    """None if option structure D is readably different, at its delivered panel
    size, from every 0/90/180/270-degree view of every structure the reference
    allows; otherwise the reason."""
    H, rgb = ref.H, ref.rgb
    Dl = D.tolist()
    cam = camera(Dl, opt_px, OPT_MARGIN)
    dimg = render_rgb(Dl, cam, rgb)
    target = _stats([tuple(c) for c in np.argwhere(D > 0)])
    for k in range(4):
        where = _rot_where(H.shape, k)
        R = np.rot90(H, k)
        offsets = _alignments(ref, k, D)
        if offsets is None:
            return f"more than {OPTIONAL_LIMIT} hidden columns could be emptied; alignment search refused"
        for off in offsets:
            comps = readable_mask(render_rgb(R.tolist(), cam, rgb, offset=off), dimg)
            r_occ = [(int(r) + off[1], int(c) + off[0]) for r, c in np.argwhere(R > 0)]
            if not comps.any() and _stats(r_occ) == target:
                return f"indistinguishable from the reference viewed at {90 * k} deg"
            lab, n = ndimage.label(comps, structure=np.ones((3, 3)))
            reach = [set() for _ in range(n)]
            for (j, i), vals in ref.domain.items():
                r, c = where[(j, i)]
                x0, y0, x1, y1 = column_window(cam, c + off[0], r + off[1], max(vals))
                for comp in np.unique(lab[y0:y1, x0:x1]):
                    if comp:
                        reach[comp - 1].add((j, i))
            if any(not r for r in reach):
                continue  # a readable difference that no hidden column can touch
            cells = sorted(set().union(*reach)) if n else sorted(ref.domain)
            total = int(np.prod([len(ref.domain[c]) for c in cells])) if cells else 1
            if total > COMBO_LIMIT:
                return f"{total} hidden-height combinations to rule out at {90 * k} deg (limit {COMBO_LIMIT})"
            for vals in itertools.product(*[ref.domain[c] for c in cells]):
                changed = {c for c, v in zip(cells, vals) if v != H[c]}
                if not changed or any(not (r & changed) for r in reach):
                    continue
                S = H.copy()
                for c, v in zip(cells, vals):
                    S[c] = v
                T = np.rot90(S, k)
                occ = [(int(r) + off[1], int(c) + off[0]) for r, c in np.argwhere(T > 0)]
                if _stats(occ) != target:
                    continue
                if not readable(render_rgb(T.tolist(), cam, rgb, offset=off), dimg) and ref.consistent(S):
                    return (f"indistinguishable from a {90 * k}-deg view of a structure the reference allows "
                            f"(cells {sorted((tuple(map(int, c)), int(S[c])) for c in changed)})")
    return None


def truth_margin(H, D, opt_px, rgb):
    """(readable px, top-face px): the smallest total area of readable difference
    components between distractor D's panel and a view of H at 0/90/180/270 deg
    drawn with D's camera and translated so the silhouette corners coincide (a
    view whose corners cannot coincide differs by a whole visible face and is
    skipped), and the area of one cube top face at that panel's scale."""
    Dl = D.tolist()
    cam = camera(Dl, opt_px, OPT_MARGIN)
    dimg = render_rgb(Dl, cam, rgb)
    target = _stats([tuple(c) for c in np.argwhere(D > 0)])
    best = None
    for k in range(4):
        R = np.rot90(H, k)
        a, b, c = _stats([tuple(x) for x in np.argwhere(R > 0)])
        s, d = target[0] - a, target[1] - b
        if (s + d) % 2 or target[2] - c != -d:
            continue
        px = int(readable_mask(render_rgb(R.tolist(), cam, rgb, offset=((s + d) // 2, (s - d) // 2)), dimg).sum())
        best = px if best is None else min(best, px)
    return best, 2 * cam["hw"] * cam["hh"] / SS ** 2


def shade_classes(img, rgb):
    """Per final pixel of an RGB render: 0 top, 1 left, 2 right (palette colour x SHADE), 3 white
    background, -1 for anything farther than FACE_TOL (L1 over RGB) from all four -- the dark
    outlines and their anti-aliased fringe."""
    refs = [[max(0, min(255, int(c * f))) for c in rgb] for f in SHADE] + [[255, 255, 255]]
    d = np.stack([np.abs(img - np.asarray(r, np.int16)).sum(axis=2) for r in refs], axis=-1)
    cls = d.argmin(axis=-1)
    cls[d.min(axis=-1) > FACE_TOL] = -1
    return cls


def face_level_mask(a, b, rgb, a_cls=None):
    """Face-level difference: pixels that both renders show as a face or the background (no outline
    pixel), of different classes, at readable contrast (max channel difference >= CONTRAST), and not
    within one pixel (8-neighbourhood) of an outline pixel in BOTH renders.  The last rule drops the
    anti-aliased fringe of an outline that sits a pixel or two apart in the two renders (that fringe
    blends into colours within FACE_TOL of a face shade); a region one render shows open still
    counts up to its outline."""
    ca = shade_classes(a, rgb) if a_cls is None else a_cls
    cb = shade_classes(b, rgb)
    k = np.ones((3, 3), bool)
    fringe = ndimage.binary_dilation(ca < 0, structure=k) & ndimage.binary_dilation(cb < 0, structure=k)
    return (ca >= 0) & (cb >= 0) & (ca != cb) & (np.abs(a - b).max(axis=2) >= CONTRAST) & ~fringe


def face_margin(ref, D, opt_px):
    """(px, top-face px, view): the smallest face-level difference (face_level_mask) between
    distractor D's panel and a 0/90/180/270-degree view of ANY structure the reference allows
    (every combination of the hidden heights in ref.domain, H included), each drawn with D's
    camera both in place and translated so that structure's own silhouette corners land on D's.

    readable_mask counts outline pixels, so an outline displaced by 1-2 px reads as a large
    difference; this measure leaves outlines out.  Pruned exactly: a structure differs from H
    only inside the windows of the hidden columns (padded PAD px beyond everything a column draws,
    more than the Lanczos support plus the one-pixel fringe rule), so where H's view at that
    translation already has >= the floor of face-level pixels outside them, no structure there can
    go below it (that count then stands in for the group).  px is exact whenever it is below MARGIN_FACE top faces.
    px is None (and view the reason) when the combinations exceed COMBO_LIMIT."""
    H, rgb = ref.H, ref.rgb
    Dl = D.tolist()
    cam = camera(Dl, opt_px, OPT_MARGIN)
    dimg = render_rgb(Dl, cam, rgb)
    dcls = shade_classes(dimg, rgb)
    face = 2 * cam["hw"] * cam["hh"] / SS ** 2
    need = MARGIN_FACE * face
    target = _stats([tuple(c) for c in np.argwhere(D > 0)])
    cells = sorted(ref.domain)
    total = int(np.prod([len(ref.domain[c]) for c in cells])) if cells else 1
    if total > COMBO_LIMIT:
        return None, face, f"{total} reference-consistent structures for the face-level margin (limit {COMBO_LIMIT})"
    structures = []
    for vals in itertools.product(*[ref.domain[c] for c in cells]):
        S = H.copy()
        for c, v in zip(cells, vals):
            S[c] = v
        if S.sum():
            structures.append((S, sorted((tuple(map(int, c)), int(v)) for c, v in zip(cells, vals) if v != H[c])))
    best, where_best = None, None
    for k in range(4):
        where = _rot_where(H.shape, k)
        R = np.rot90(H, k)
        groups = {}
        for S, changed in structures:
            T = np.rot90(S, k)
            groups.setdefault((0, 0), []).append((T, changed))
            a, b, c = _stats([tuple(x) for x in np.argwhere(T > 0)])
            s, d = target[0] - a, target[1] - b
            if (s + d) % 2 == 0 and target[2] - c == -d and (s, d) != (0, 0):
                groups.setdefault(((s + d) // 2, (s - d) // 2), []).append((T, changed))
        for off, members in groups.items():
            base = face_level_mask(dimg, render_rgb(R.tolist(), cam, rgb, offset=off), rgb, dcls)
            outside = base.copy()
            for (j, i), vals in ref.domain.items():
                r, c = where[(j, i)]
                x0, y0, x1, y1 = column_window(cam, c + off[0], r + off[1], max(vals))
                outside[y0:y1, x0:x1] = False
            if int(outside.sum()) >= need:
                px, view = int(outside.sum()), "bound"
            else:
                px, view = None, None
                for T, changed in members:
                    if np.array_equal(T, R):
                        fl = int(base.sum())
                    else:
                        fl = int(face_level_mask(dimg, render_rgb(T.tolist(), cam, rgb, offset=off), rgb, dcls).sum())
                    if px is None or fl < px:
                        px, view = fl, changed
            if best is None or px < best:
                what = ("every allowed structure (lower bound outside the hidden columns)" if view == "bound"
                        else "the reference" if not view else f"the structure with hidden cells {view}")
                best, where_best = px, f"a {90 * k}-deg view of {what}" + (" (translated)" if off != (0, 0) else "")
    return best, face, where_best


# --------------------------------------------------------------------------- tier rules
# Own copy of the documented tier recipe (TIER_DEFINITIONS): rung per tier and
# index parity, grid, cube count, height cap, yaw set, number of mirror-image
# distractors and the edit that makes the other distractors.  L1 lists
# its floors instead of a grid range: {(rows, cols): (min, max cubes)}, and the
# reference's occupied bounding box must be the whole floor (inner rows and
# columns may be empty).
RUNG_RULES = {
    "L1": dict(tier="easy", parity=None, size=448, rows=(3, 5), cols=(4, 5), cubes=(4, 9), hmax=2,
               yaws=(90, 270), mirrors=None, edit="cells", n=(2, 3),
               floors={(5, 5): (4, 8), (4, 4): (9, 9), (3, 5): (9, 9)}),
    "L2": dict(tier="medium", parity=0, size=448, rows=(2, 3), cols=(2, 3), cubes=(5, 8), hmax=3,
               yaws=(90, 180, 270), mirrors=1, edit="cells", n=(2, 2)),
    "L3": dict(tier="medium", parity=1, size=448, rows=(3, 3), cols=(3, 4), cubes=(7, 10), hmax=3,
               yaws=(90, 180, 270), mirrors=1, edit="move", n=(1, 2)),
    "L4": dict(tier="hard", parity=0, size=448, rows=(3, 3), cols=(3, 4), cubes=(7, 12), hmax=3,
               yaws=(90, 180, 270), mirrors=1, edit="move", n=(1, 1)),
    "L5": dict(tier="hard", parity=1, size=448, rows=(3, 4), cols=(3, 4), cubes=(10, 14), hmax=4,
               yaws=(90, 180, 270), mirrors=1, edit="move", n=(1, 1)),
    "L6": dict(tier="xhard", parity=None, size=588, rows=(4, 4), cols=(5, 5), cubes=(20, 28), hmax=3,
               yaws=(90, 180, 270), mirrors=1, edit="move6", n=(1, 1)),
    "L7": dict(tier="xxhard", parity=None, size=588, rows=(5, 5), cols=(5, 5), cubes=(30, 42), hmax=4,
               yaws=(90, 180, 270), mirrors=1, edit="swap", n=(1, 1)),
}
BASE_PALETTES = [(205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160)]


def _edit_ok(H, D, rule):
    """D is, under some yaw, H's grid with the rung's edit applied."""
    lo, hi = rule["n"]
    for k in range(4):
        R = np.rot90(D, k)
        if R.shape != H.shape or R.max() > rule["hmax"]:
            continue
        diff = R - H
        cells = [tuple(c) for c in np.argwhere(diff != 0)]
        if rule["edit"] == "cells" and lo <= len(cells) <= hi:
            return True
        if rule["edit"] == "move" and diff.sum() == 0 and int(np.abs(diff).sum()) in {2 * m for m in range(lo, hi + 1)}:
            return True
        if rule["edit"] == "move6" and len(cells) == 2 and diff.sum() == 0 and np.abs(diff).max() == 1:
            src, dst = (cells if diff[cells[0]] < 0 else cells[::-1])
            if H[src] > 1 and H[dst] < 3:
                return True
        if rule["edit"] == "swap" and len(cells) == 2 and R[cells[0]] == H[cells[1]] and R[cells[1]] == H[cells[0]]:
            return True
    return False


def tier_rule_problems(record, H, opts, gold, rgb):
    m = record.get("meta") or {}
    rung, tier = m.get("rung"), record.get("tier")
    rule = RUNG_RULES.get(rung)
    if rule is None or rule["tier"] != tier:
        return [f"tier rule: rung {rung!r} does not belong to tier {tier!r}"]
    probs = []
    index = (record.get("source") or {}).get("index")
    if rule["parity"] is not None and (not isinstance(index, int) or index % 2 != rule["parity"]):
        probs.append(f"tier rule: {rung} is used on {'even' if rule['parity'] == 0 else 'odd'} indices, got {index!r}")
    if int(m.get("size", 0)) != rule["size"]:
        probs.append(f"tier rule: {tier} images are {rule['size']} px")
    if tuple(rgb) not in (BASE_PALETTES if rule["size"] == 448 else BASE_PALETTES[:1]):
        probs.append(f"tier rule: palette {tuple(rgb)} is not one of the tier's palettes")
    N, M = H.shape
    cubes = rule["cubes"]
    if rule.get("floors"):
        # L1: one of the listed floor grids, an occupied bounding box that is the whole floor (cubes in
        # its first and last row and column; inner rows/columns may be empty), and that floor's cube bound
        if (N, M) not in rule["floors"]:
            probs.append(f"tier rule: {rung} grid {N}x{M} is not one of the floors "
                         f"{', '.join(f'{a}x{b}' for a, b in rule['floors'])}")
        else:
            cubes = rule["floors"][(N, M)]
        occ = np.argwhere(H > 0)
        span = tuple(int(x) for x in occ.max(0) - occ.min(0) + 1) if len(occ) else (0, 0)
        if span != (N, M):
            probs.append(f"tier rule: {rung} occupied bounding box {span[0]}x{span[1]} is not the whole {N}x{M} floor")
    elif not (rule["rows"][0] <= N <= rule["rows"][1] and rule["cols"][0] <= M <= rule["cols"][1]):
        probs.append(f"tier rule: {rung} grid {N}x{M} outside {rule['rows']} x {rule['cols']}")
    if not cubes[0] <= int(H.sum()) <= cubes[1] or H.max() > rule["hmax"]:
        probs.append(f"tier rule: {rung} reference has {int(H.sum())} cubes, height {int(H.max())}; "
                     f"allowed {tuple(cubes)}, <= {rule['hmax']}")
    if m.get("yaw_degrees") not in rule["yaws"]:
        probs.append(f"tier rule: yaw {m.get('yaw_degrees')} not in {rule['yaws']}")
    mirror = canon(H[:, ::-1])
    kinds = list(m.get("option_kinds") or [])
    if len(kinds) != 4 or kinds[gold] != "truth":
        probs.append("tier rule: option_kinds must mark the matching option as the truth")
    n_mirror = 0
    for n, O in enumerate(opts):
        if n == gold:
            continue
        is_mirror = canon(O) == mirror
        n_mirror += is_mirror
        kind = kinds[n] if len(kinds) == 4 else None
        if kind == "mirror" and not is_mirror:
            probs.append(f"tier rule: option {LETTERS[n]} is labelled mirror but is not the mirror image")
        if rule["mirrors"] is not None and is_mirror:
            if kind != "mirror":
                probs.append(f"tier rule: option {LETTERS[n]} is the mirror image but labelled {kind!r}")
            continue
        if not _edit_ok(H, O, rule):
            probs.append(f"tier rule: option {LETTERS[n]} is not a {rung} edit ({rule['edit']} {rule['n']}) "
                         f"of the reference under any yaw")
    if rule["mirrors"] is not None and n_mirror != rule["mirrors"]:
        probs.append(f"tier rule: {rung} has {rule['mirrors']} mirror-image distractor(s), found {n_mirror}")
    return probs


# --------------------------------------------------------------------------- entry point
def verify(record, image):
    probs = []
    question = record.get("question")
    if question != QUESTION:
        probs.append("question text differs from the declared form")
    shown = format_example_letters(str(question or ""))
    if shown:
        probs.append(f"the question shows example answer letter(s) {shown}; an example must never be an answer")
    if record.get("answer_type") != "choice" or record.get("choices") != LETTERS:
        probs.append("answer_type/choices must be choice/[A,B,C,D]")
    ans = record.get("answer")
    if ans not in LETTERS:
        return probs + [f"answer {ans!r} is not one of A-D"]
    if ans in shown:
        probs.append(f"the gold letter {ans} appears in the question as an example")
    m = record.get("meta") or {}
    try:
        H = np.asarray(m["heightmap"], dtype=int)
        opts = [np.asarray(o, dtype=int) for o in m["option_heightmaps"]]
        size = int(m["size"])
        rgb = tuple(int(v) for v in m["palette_rgb"])
    except Exception as exc:  # noqa: BLE001
        return probs + [f"meta incomplete: {exc!r}"]
    if H.ndim != 2 or H.min() < 0 or H.sum() == 0 or len(opts) != 4 or len(rgb) != 3:
        return probs + ["meta heightmap/options/palette malformed"]
    if any(o.ndim != 2 or o.min() < 0 or o.sum() == 0 for o in opts):
        return probs + ["option heightmap malformed"]
    if image.size != (size, size):
        return probs + [f"image is {image.size}, meta size {size}"]
    ref_box, opt_boxes, slots = layout(size)
    if m.get("reference_px") != ref_box[2] or m.get("option_px") != opt_boxes[0][2]:
        probs.append("meta panel sizes disagree with the layout")

    # 2. decode the image against meta: the five panels, then everything outside them
    arr = np.asarray(image.convert("RGB"))
    probs += _decode_panel(arr, ref_box, H.tolist(), REF_MARGIN, True, rgb, "reference")
    for n, (box, O) in enumerate(zip(opt_boxes, opts)):
        probs += _decode_panel(arr, box, O.tolist(), OPT_MARGIN, False, rgb, f"option {LETTERS[n]}")
    probs += chrome_problems(arr, size)
    if probs:
        return probs

    # 3. correctness: the gold letter from the structures alone
    yaws = [yaw_of(H, O) for O in opts]
    matches = [n for n, k in enumerate(yaws) if k is not None]
    if len(matches) != 1:
        return [f"{len(matches)} options are rotations of the reference structure"]
    gold = LETTERS[matches[0]]
    if yaws[matches[0]] == 0:
        probs.append("the matching option is the reference seen from the same direction")
    if gold != ans:
        probs.append(f"answer {ans} but the rotated reference is option {gold}")
    if m.get("correct_index") != matches[0] or m.get("yaw_degrees") != 90 * yaws[matches[0]]:
        probs.append("meta correct_index/yaw_degrees disagree with the structures")
    if len({canon(o) for o in opts}) != 4:
        probs.append("two options show the same structure")
    if canon(H[:, ::-1]) == canon(H):
        probs.append("reference is achiral: its mirror image is also a rotation")
    e = extent(H)
    if any(np.rot90(e, k).shape == e.shape and np.array_equal(np.rot90(e, k), e) for k in (1, 2, 3)):
        probs.append("reference has rotational self-symmetry")
    if probs:
        return probs

    # 4. uniqueness at the delivered resolution
    ref = Reference(H, ref_box[2], rgb)
    if ref.joint:
        return [f"reference is ambiguous at the delivered resolution: {ref.joint[0]}"
                + (f" (+{len(ref.joint) - 1} more)" if len(ref.joint) > 1 else "")]
    for n, O in enumerate(opts):
        if n == matches[0]:
            continue
        why = distractor_explained(ref, O, opt_boxes[n][2])
        if why:
            probs.append(f"option {LETTERS[n]} could also be correct: {why}")
            continue
        # 5. margin: more than a readable speck separates the distractor from the truth
        px, face = truth_margin(H, O, opt_boxes[n][2], rgb)
        if px is not None and px < MARGIN_FACE * face:
            probs.append(f"option {LETTERS[n]} differs from an aligned view of the reference by only {px} readable "
                         f"px (< {MARGIN_FACE} x the {face:.0f} px cube top face)")
            continue
        # 5b. face-level margin against every view of every structure the reference allows
        fl, face, view = face_margin(ref, O, opt_boxes[n][2])
        if fl is None:
            probs.append(f"option {LETTERS[n]}: {view}; face-level margin search refused")
        elif fl < MARGIN_FACE * face:
            probs.append(f"option {LETTERS[n]} differs from {view} by only {fl} face-level px (< {MARGIN_FACE} x the "
                         f"{face:.0f} px cube top face; outline pixels excluded)")

    # 6. the tier's stated recipe (records written by the release driver carry their tier)
    if "tier" in record:
        probs += tier_rule_problems(record, H, opts, matches[0], rgb)
    return probs
