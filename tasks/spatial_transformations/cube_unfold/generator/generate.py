"""Base generator of cube_unfold, in the style of BabyVision's cube-unfolding task.

LEVEL: high (3D / mental-simulation tier). The bottleneck is folding a flat net
into a cube in the mind's eye and matching it against isometric corner views.

A valid hexomino cube net is chosen (one of the 11), each of its 6 squares gets a
distinct symbol (a solid color, or an orientation-sensitive arrow/glyph). The net
is FOLDED by BFS, tracking each square's 3D outward NORMAL and a 3D UP vector.
That yields the cube's exact face adjacency, opposite-face pairs, and per-face
symbol orientation.

We then render N candidate cubes as isometric corner views (each shows 3 faces
meeting at the top-front corner). Some candidates are real views of the folded
cube (consistent), some are tampered (swapped faces -> wrong adjacency, replaced
face -> a symbol that isn't even on the cube or is in the wrong place, or rotated
symbol -> wrong orientation). The question asks which candidates match the net;
the answer is the comma-separated list of consistent indices, e.g. "2,5,7".

CORRECTNESS (exact, no heuristics)
----------------------------------
The folded cube is a map face_id -> (normal, up). The set of ALL valid corner
views is computed by applying each of the 24 cube rotations and reading the three
front-top faces (normals +Z top, +Y right, +X left) into a signature
  { 'top':(face_id, up_step), 'right':(...), 'left':(...) }.
A candidate is CONSISTENT iff its own signature is a member of those 24. This is
bulletproof: it compares against every legal view rather than a hand rule. When
symbols are rotation-INVARIANT (solid color / centred dot) the up_step is dropped
from the signature so adjacency alone decides correctness.

Difficulty knob: number of candidates, orientation-sensitive symbols vs plain
colors, and how subtle the tampering is.

Reproducible: each problem is a pure function of (seed, idx).
"""
from __future__ import annotations

import math
import random
from collections import deque
from itertools import combinations, permutations, product

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing.fonts import load_bundled_font


def derive_rng(seed: int, idx: int) -> random.Random:
    return random.Random(f"{seed}:{idx}")


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def neg(a):
    return (-a[0], -a[1], -a[2])


def rot90(v, axis, k=1):
    """Rotate integer vector v by 90*k degrees about integer unit axis."""
    th = math.pi / 2 * k
    ax = np.array(axis, float)
    ax = ax / np.linalg.norm(ax)
    vv = np.array(v, float)
    out = (vv * math.cos(th) + np.cross(ax, vv) * math.sin(th)
           + ax * np.dot(ax, vv) * (1 - math.cos(th)))
    return tuple(int(round(x)) for x in out)


# ---------------------------------------------------------------------------
# The 11 valid hexomino cube nets (enumerated + folded-validated at import).
# ---------------------------------------------------------------------------
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def fold_net(cells):
    """BFS-fold a net (list of (r,c)) -> {face_index: (normal, up)}.

    Square 0 is the top (normal +Z, up +Y = north). Each grid step folds the
    neighbour 90 deg around the shared edge; the neighbour's normal becomes the
    in-plane step direction and its up is the parent's up rotated the same way."""
    cells = [tuple(cell) for cell in cells]
    cellset = set(cells)
    start = cells[0]
    frames = {start: ((0, 0, 1), (0, 1, 0))}
    q = deque([start])
    while q:
        cur = q.popleft()
        nC, uC = frames[cur]
        rightC = cross(uC, nC)
        plane = {"E": rightC, "W": neg(rightC), "N": uC, "S": neg(uC)}
        for dn, (dr, dc) in DIRS.items():
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in cellset and nb not in frames:
                move = plane[dn]
                hinge = cross(nC, move)
                nu = None
                for k in (1, -1):
                    if rot90(nC, hinge, k) == move:
                        nu = rot90(uC, hinge, k)
                        break
                frames[nb] = (move, nu)
                q.append(nb)
    return {i: frames[c] for i, c in enumerate(cells)}


def is_valid_net(cells) -> bool:
    fr = fold_net(cells)
    return len(fr) == 6 and len({fr[i][0] for i in fr}) == 6


def _enumerate_valid_nets():
    """Grow all 6-cell polyominoes, keep the 11 distinct ones that fold to a cube."""
    def norm(cs):
        mr = min(r for r, c in cs)
        mc = min(c for r, c in cs)
        return frozenset((r - mr, c - mc) for r, c in cs)

    frontier = {frozenset([(0, 0)])}
    for _ in range(5):
        nf = set()
        for shape in frontier:
            for (r, c) in shape:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb not in shape:
                        s = set(shape)
                        s.add(nb)
                        nf.add(norm(s))
        frontier = nf

    def canon(cs):
        forms = []
        pts = list(cs)
        for _ in range(4):
            pts = [(c, -r) for r, c in pts]
            for refl in (pts, [(r, -c) for r, c in pts]):
                mr = min(r for r, c in refl)
                mc = min(c for r, c in refl)
                forms.append(frozenset((r - mr, c - mc) for r, c in refl))
        return min(forms, key=lambda f: tuple(sorted(f)))

    # Iterate in sorted order so the 11 drawings and their cell order do not
    # depend on set iteration order (hash details of the running interpreter).
    uniq = {}
    for shape in sorted(frontier, key=lambda f: tuple(sorted(f))):
        cells = sorted(shape)
        if is_valid_net(cells):
            uniq.setdefault(canon(cells), cells)
    return [uniq[k] for k in sorted(uniq, key=lambda f: tuple(sorted(f)))]


VALID_NETS = _enumerate_valid_nets()
assert len(VALID_NETS) == 11, f"expected 11 nets, got {len(VALID_NETS)}"


# ---------------------------------------------------------------------------
# 24 cube rotations + the set of all valid corner-view signatures
# ---------------------------------------------------------------------------
def _rotation_matrices():
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if round(np.linalg.det(M)) == 1:
                mats.append(M)
    return mats


ROTS = _rotation_matrices()  # 24

GENERATOR_VERSION = "physical-fold"
# Right-handed coordinates viewed from (+X,+Y,+Z). Screen-right is +Y-X.
SLOT_NORMAL = {"top": (0, 0, 1), "right": (0, 1, 0), "left": (1, 0, 0)}
REF_UP = {(0, 0, 1): (0, 1, 0), (1, 0, 0): (0, 0, 1), (0, 1, 0): (0, 0, 1),
          (0, 0, -1): (0, 1, 0), (-1, 0, 0): (0, 0, 1), (0, -1, 0): (0, 0, 1)}


def _up_step(nrm, u):
    """How many 90-deg right-hand turns about nrm take REF_UP[nrm] -> u (0..3)."""
    cur = np.array(REF_UP[nrm])
    for k in range(4):
        if tuple(int(x) for x in cur) == tuple(u):
            return k
        cur = np.array(cross(nrm, cur))
    return None


def apply_rot(M, frames):
    out = {}
    for fid, (n, u) in frames.items():
        out[fid] = (tuple(int(x) for x in (M @ np.array(n))),
                    tuple(int(x) for x in (M @ np.array(u))))
    return out


def view_signature(frames, oriented: bool):
    """{slot: (face_id, up_step or 0)} for the 3 front-top faces, or None."""
    byn = {n: (fid, u) for fid, (n, u) in frames.items()}
    sig = {}
    for slot, nrm in SLOT_NORMAL.items():
        if nrm not in byn:
            return None
        fid, u = byn[nrm]
        sig[slot] = (fid, _up_step(nrm, u) if oriented else 0)
    return sig


def all_valid_views(folded, oriented: bool):
    """Set of every corner-view signature reachable by the 24 rotations."""
    out = set()
    for M in ROTS:
        s = view_signature(apply_rot(M, folded), oriented)
        if s:
            out.add(_sig_key(s))
    return out


def _sig_key(sig):
    return tuple((slot, sig[slot]) for slot in ("top", "right", "left"))


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
# Palette: colour is the ONLY face identity in easy/medium, so near-duplicate
# faces would flip candidates.  Every palette below keeps all 15 face pairs at least
# MIN_PALETTE_DE00 apart, keeps the white glyph visible on every face (WCAG
# contrast >= 1.9) and keeps faces away from the near-black outline.
COLOR_SETS = [
    # red, blue, green, gold, cyan, magenta            (min dE00 29.2)
    [(214, 39, 40), (40, 80, 200), (40, 160, 60), (220, 180, 30),
     (0, 170, 210), (210, 40, 160)],
    # crimson, orange, teal, navy, lime, pink          (min dE00 29.4)
    [(180, 20, 50), (245, 130, 20), (0, 150, 140), (30, 50, 130),
     (130, 190, 30), (240, 100, 170)],
    # amber, dark green, sky, brown, purple, lime      (min dE00 31.0)
    [(240, 170, 0), (20, 100, 40), (60, 150, 230), (120, 70, 40),
     (130, 50, 170), (130, 190, 30)],
]
MONOCHROME = (65, 85, 112)   # xxhard: every face this colour, identity by glyph only
MIN_PALETTE_DE00 = 25.0
MIN_GLYPH_CONTRAST = 1.9     # WCAG contrast ratio of the white glyph on a face
OUTLINE = (20, 20, 20)
# orientation-sensitive glyphs (clearly change under rotation)
ORIENT_GLYPHS = ["arrow", "tri", "ell", "tee", "flag", "key"]


def _srgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ c / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def ciede2000(rgb1, rgb2):
    """CIEDE2000 colour difference of two sRGB triples (D65)."""
    L1, a1, b1 = _srgb_to_lab(rgb1)
    L2, a2, b2 = _srgb_to_lab(rgb2)
    cbar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dlp, dcp = L2 - L1, c2p - c1p
    dhp = 0.0 if c1p * c2p == 0 else h2p - h1p
    if dhp > 180:
        dhp -= 360
    elif dhp < -180:
        dhp += 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2))
    lbp, cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + (360 if h1p + h2p < 360 else -360)) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dth = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    rc = 2 * math.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / math.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -math.sin(math.radians(2 * dth)) * rc
    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2
                     + rt * (dcp / sc) * (dHp / sh))


def _luminance(rgb):
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def glyph_contrast(rgb):
    return 1.05 / (_luminance(rgb) + 0.05)


def palette_min_de00(palette):
    return min(ciede2000(a, b) for i, a in enumerate(palette) for b in palette[i + 1:])


for _pal in COLOR_SETS:
    assert len(_pal) == 6 and len(set(_pal)) == 6
    assert palette_min_de00(_pal) >= MIN_PALETTE_DE00, (_pal, palette_min_de00(_pal))
    for _c in _pal:
        assert glyph_contrast(_c) >= MIN_GLYPH_CONTRAST, _c
        assert ciede2000(_c, OUTLINE) >= 20 and min(_c) <= 170, _c
assert glyph_contrast(MONOCHROME) >= MIN_GLYPH_CONTRAST


# ---------------------------------------------------------------------------
# Isometric rendering
# ---------------------------------------------------------------------------
def _iso(p, a, b, c, ox, oy):
    x, y, z = p
    return (ox + (y - x) * a, oy + (x + y) * b - z * c)


def face_frame(slot, step):
    """Outward face frame; rotate in 3D before projecting either in-plane axis."""
    normal = SLOT_NORMAL[slot]
    up = rot90(REF_UP[normal], normal, step)
    return cross(up, normal), up, normal


def draw_glyph(draw, slot_pts, right_2d, up_2d, kind, sym, oriented):
    """Affine-project the entire face-local artwork, including circles/strokes.

    The two basis vectors are projected *unit face edges*. Do not normalize
    them or replace right with a screen perpendicular: either loses the face's
    foreshortening and can mirror asymmetric artwork.
    """
    color = tuple(sym["color"])
    draw.polygon(slot_pts, fill=color)
    cx = sum(p[0] for p in slot_pts) / 4
    cy = sum(p[1] for p in slot_pts) / 4
    scale = (0.44 if sym.get("large_glyph") else 0.38) if oriented else 0.22

    def L(x, y):
        return (cx + scale * (x * right_2d[0] + y * up_2d[0]),
                cy + scale * (x * right_2d[1] + y * up_2d[1]))

    def polygon(points, fill=(255, 255, 255)):
        draw.polygon([L(x, y) for x, y in points], fill=fill)

    def circle(x, y, radius, fill=(255, 255, 255)):
        polygon([(x + radius * math.cos(t * math.tau / 64),
                  y + radius * math.sin(t * math.tau / 64))
                 for t in range(64)], fill)

    def stroke(points, width=0.18):
        if sym.get("large_glyph"):
            width = max(width, 0.24)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            dx = -(y1 - y0) * width / (2 * length)
            dy = (x1 - x0) * width / (2 * length)
            polygon([(x0 + dx, y0 + dy), (x1 + dx, y1 + dy),
                     (x1 - dx, y1 - dy), (x0 - dx, y0 - dy)])
        for x, y in points:
            circle(x, y, width / 2)

    if not oriented:
        circle(0, 0, 1)
        return
    kind = sym.get("glyph", kind)
    if kind == "arrow":
        stroke([(0, -0.85), (0, 0.6)])
        polygon([(0, 0.95), (-0.48, 0.30), (0.48, 0.30)])
    elif kind == "tri":
        polygon([(0, 0.9), (-0.8, -0.7), (0.8, -0.7)])
    elif kind == "ell":
        stroke([(-0.5, 0.85), (-0.5, -0.7), (0.7, -0.7)], 0.28)
    elif kind == "tee":
        stroke([(-0.75, 0.7), (0.75, 0.7)], 0.28)
        stroke([(0, 0.7), (0, -0.8)], 0.28)
    elif kind == "flag":
        stroke([(-0.55, 0.8), (-0.55, -0.85)], 0.24)
        polygon([(-0.5, 0.85), (0.7, 0.55), (-0.5, 0.25)])
    elif kind == "key":
        circle(0, 0.4, 0.45)
        circle(0, 0.4, 0.26, color)
        stroke([(0, -0.05), (0, -0.85)])
        stroke([(0, -0.5), (0.5, -0.5)])
    else:
        raise ValueError(f"unknown face glyph: {kind}")


def cube_projection(size):
    """(ss, a, b, c, ox, oy) of the isometric camera on the supersampled canvas."""
    ss = 3
    canvas = size * ss
    pad = int(size * 0.16) * ss
    a = (canvas - 2 * pad) * 0.5
    return ss, a, a * 0.5, a * 1.04, canvas / 2, canvas * 0.62


def render_cube(sig, symbols, glyph_kind, oriented, size):
    """Render the outer faces with the same right-handed frame used by folding."""
    ss, a, b, c, ox, oy = cube_projection(size)
    canvas = size * ss
    project = lambda p: _iso(p, a, b, c, ox, oy)
    origin = np.asarray(project((0, 0, 0)))
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    outlines = []
    for slot in ("top", "right", "left"):
        fid, step = sig[slot]
        right, up, normal = face_frame(slot, step if oriented else 0)
        center = (np.ones(3) + normal) / 2
        quad = [center + x * np.asarray(right) + y * np.asarray(up)
                for x, y in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]]
        pts = [project(p) for p in quad]
        right_2d = np.asarray(project(right)) - origin
        up_2d = np.asarray(project(up)) - origin
        draw_glyph(d, pts, right_2d, up_2d, glyph_kind, symbols[fid], oriented)
        outlines.append(pts)
    for pts in outlines:
        d.line(pts + [pts[0]], fill=(20, 20, 20), width=2 * ss)
    return img.resize((size, size), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Net drawing (the unfolded diagram)
# ---------------------------------------------------------------------------
def render_net(cells, symbols, glyph_kind, oriented, cell_px):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    R, C = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    mr, mc = min(rs), min(cs)
    width, height = C * cell_px + 4, R * cell_px + 4
    ss = 3
    img = Image.new("RGBA", (width * ss, height * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for fid, (r, c) in enumerate(cells):
        x0, y0 = ((c - mc) * cell_px + 2) * ss, ((r - mr) * cell_px + 2) * ss
        edge = cell_px * ss
        quad = [(x0, y0), (x0 + edge, y0), (x0 + edge, y0 + edge), (x0, y0 + edge)]
        draw_glyph(d, quad, (edge, 0), (0, -edge), glyph_kind, symbols[fid], oriented)
        d.line(quad + [quad[0]], fill=(20, 20, 20), width=2 * ss)
    return img.resize((width, height), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Candidate (cube view) generation
# ---------------------------------------------------------------------------
def tamper(sig, valid_keys, n_faces_present, oriented, rng):
    """Return a corrupted signature whose key is NOT in valid_keys, or None."""
    slots = ("top", "right", "left")
    modes = ["swap", "replace"]
    if oriented:
        modes += ["rotate", "rotate"]
    rng.shuffle(modes)
    for mode in modes:
        new = dict(sig)
        if mode == "swap":
            s1, s2 = rng.sample(slots, 2)
            new[s1], new[s2] = sig[s2], sig[s1]
        elif mode == "replace":
            s = rng.choice(slots)
            shown = {sig[sl][0] for sl in slots}     # avoid duplicating a face
            others = [f for f in range(n_faces_present) if f not in shown]
            if not others:
                continue
            new[s] = (rng.choice(others), sig[s][1])
        else:  # rotate one symbol's orientation
            s = rng.choice(slots)
            fid, st = sig[s]
            new[s] = (fid, (st + rng.choice([1, 2, 3])) % 4)
        if _sig_key(new) not in valid_keys:
            return new
    return None


# large      large glyphs (scale 0.44, strokes >= 0.24 face units) and the compact
#            "extended" page layout that gives the candidate grid more height
# readable   cube_readable must pass at the delivered size (every single-slot
#            glyph turn and every candidate pair differ by a visible patch)
# Hard is large + readable at 588 px like xhard/xxhard: at 448 px most hard items
# (8-9 cubes of about 80 px with 0.38-scale glyphs) fail cube_readable, so glyph
# orientation -- and the gold -- could not be recovered from the image.
DIFFICULTY = {
    "easy":   dict(n_cand=(4, 4), oriented=False),
    "medium": dict(n_cand=(6, 6), oriented=False),
    "hard":   dict(n_cand=(8, 9), oriented=True, large=True, readable=True),
    "xhard":  dict(n_cand=(10, 10), oriented=True, rotation_only=True, large=True, readable=True),
    "xxhard": dict(n_cand=(12, 12), oriented=True, rotation_only=True, monochrome=True,
                   large=True, readable=True),
}
def build_problem(rng, difficulty):
    cfg = DIFFICULTY[difficulty]
    cells = list(rng.choice(VALID_NETS))
    rng.shuffle(cells)  # randomise which square is the BFS start (face labels)
    folded = fold_net(cells)

    oriented = cfg["oriented"]
    palette = rng.randrange(len(COLOR_SETS))
    cols = list(COLOR_SETS[palette])
    rng.shuffle(cols)
    if cfg.get("monochrome"):
        palette = -1
        cols = [MONOCHROME] * 6
    glyph_kind = "mixed" if oriented else "dot"
    if oriented:
        glyphs = list(ORIENT_GLYPHS)
        rng.shuffle(glyphs)
        symbols = [{"color": cols[i], "glyph": glyphs[i], "large_glyph": bool(cfg.get("large"))}
                   for i in range(6)]
    else:
        symbols = [{"color": cols[i]} for i in range(6)]

    valid_keys = all_valid_views(folded, oriented)
    valid_sigs = []
    for M in ROTS:
        s = view_signature(apply_rot(M, folded), oriented)
        if s:
            valid_sigs.append(s)

    n_cand = rng.randint(*cfg["n_cand"])
    n_good = rng.randint(max(1, n_cand // 4), max(2, n_cand // 2))

    candidates = []          # list of (sig, is_consistent)
    used_keys = set()
    # positives: distinct valid views
    rng.shuffle(valid_sigs)
    for s in valid_sigs:
        if len([c for c in candidates if c[1]]) >= n_good:
            break
        k = _sig_key(s)
        if k in used_keys:
            continue
        used_keys.add(k)
        candidates.append((s, True))
    # negatives: tamper distinct valid views
    rng.shuffle(valid_sigs)
    tries = 0
    while len(candidates) < n_cand and tries < 400:
        tries += 1
        base = rng.choice(valid_sigs)
        if cfg.get("rotation_only"):
            bad = dict(base)
            slot = rng.choice(("top", "right", "left"))
            face, step = bad[slot]
            bad[slot] = (face, (step + rng.choice((1, 2, 3))) % 4)
        else:
            bad = tamper(base, valid_keys, 6, oriented, rng)
        if bad is None:
            continue
        k = _sig_key(bad)
        if k in used_keys or k in valid_keys:   # explicit for rotation_only too
            continue
        used_keys.add(k)
        candidates.append((bad, False))

    rng.shuffle(candidates)
    consistent_idx = [i + 1 for i, (s, ok) in enumerate(candidates) if ok]
    # Guard: at least one consistent and at least one inconsistent
    if len(candidates) != n_cand or not consistent_idx or len(consistent_idx) == len(candidates):
        return None

    answer = ",".join(str(i) for i in consistent_idx)
    return dict(cells=cells, folded=folded, symbols=symbols, glyph_kind=glyph_kind,
                oriented=oriented, candidates=candidates, palette=palette,
                consistent_idx=consistent_idx, answer=answer,
                difficulty=difficulty, valid_count=len(valid_keys))


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
def _page_layout(prob, size):
    """Page geometry shared by render_page, page_geometry and cube_readable."""
    cells = prob["cells"]
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    R = max(rs) - min(rs) + 1; C = max(cs) - min(cs) + 1
    margin = int(size * 0.025)
    extended = bool(DIFFICULTY.get(prob.get("difficulty"), {}).get("large"))
    answer_h = int(size * (0.035 if extended else 0.06))          # reserved strip at the very bottom
    label_h = int(size * (0.035 if extended else 0.045))          # space under each cube for its number

    net_area_h = int(size * (0.26 if extended else 0.30))
    net_area_w = int(size * 0.86)
    cell_px = int(min(net_area_w / C, net_area_h / R))
    net_w = C * cell_px + 4
    net_h = R * cell_px + 4
    ny = int(size * 0.025)

    n = len(prob["candidates"])
    cols = min(4, n)
    rows = math.ceil(n / cols)
    grid_top = ny + net_h + int(size * (0.025 if extended else 0.045))
    grid_bottom = size - answer_h
    grid_h = grid_bottom - grid_top
    cell_w = (size - 2 * margin) / cols
    cell_h = grid_h / rows
    cube_size = int(min(cell_w, cell_h - label_h) * 0.98)
    return dict(R=R, C=C, cell_px=cell_px, net_w=net_w, net_h=net_h, ny=ny,
                cols=cols, rows=rows, grid_top=grid_top, cell_w=cell_w,
                cell_h=cell_h, cube_size=cube_size, margin=margin,
                label_h=label_h, answer_h=answer_h)


def _cube_xy(L, i, size):
    r, c = i // L["cols"], i % L["cols"]
    cx = int(L["margin"] + c * L["cell_w"] + (L["cell_w"] - L["cube_size"]) / 2)
    cy = int(L["grid_top"] + r * L["cell_h"])
    return cx, cy


def page_geometry(prob, size):
    """Where everything is drawn, in final-image pixels (x right, y down; a pixel
    i covers [i, i+1)).  Lets a checker locate every face without this module.

    net:   cell (r, c) covers x in [x0 + (c - c_min) * cell_px, + cell_px] and
           y in [y0 + (r - r_min) * cell_px, + cell_px]; drawn with right = +x,
           up = -y (the printed side, viewed from the front).
    cubes: candidate i (1-based, row-major) is a cube_size square whose top-left
           is cubes[i-1]; a 3D point (x, y, z) of the unit cube [0,1]^3 is drawn
           at (left + ox + (y - x) * a, top + oy + (x + y) * b - z * c).
    labels: the number i is drawn with font `font` at `label_px` px, its text
           origin at labels[i-1] (PIL ImageDraw.text anchor 'la').
    """
    L = _page_layout(prob, size)
    cells = prob["cells"]
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    nx = (size - L["net_w"]) // 2
    ss, a, b, c, ox, oy = cube_projection(L["cube_size"])
    label_px = max(13, size // 30)
    label_font = load_bundled_font(label_px, "sans-bold")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cubes, labels = [], []
    for i in range(len(prob["candidates"])):
        cx, cy = _cube_xy(L, i, size)
        cubes.append([cx, cy])
        bb = probe.textbbox((0, 0), str(i + 1), font=label_font)
        labels.append([cx + L["cube_size"] / 2 - (bb[2] - bb[0]) / 2, cy + L["cube_size"] + 1])
    return {
        "size": size,
        "net": {"x0": nx + 2, "y0": L["ny"] + 2, "cell_px": L["cell_px"], "r_min": mr, "c_min": mc},
        "cube_size": L["cube_size"],
        "cubes": cubes,
        "iso": {"a": a / ss, "b": b / ss, "c": c / ss, "ox": ox / ss, "oy": oy / ss},
        "labels": labels,
        "label_px": label_px,
        "font": "assets/fonts/DejaVuSans-Bold.ttf",
        "background": [250, 250, 252],
        "outline": list(OUTLINE),
    }


def render_page(prob, size):
    img = Image.new("RGB", (size, size), (250, 250, 252))
    d = ImageDraw.Draw(img)
    L = _page_layout(prob, size)
    font = load_bundled_font(max(13, size // 30), "sans-bold")

    net = render_net(prob["cells"], prob["symbols"], prob["glyph_kind"],
                     prob["oriented"], L["cell_px"])
    nx = (size - net.width) // 2
    img.paste(net, (nx, L["ny"]), net)
    d.text((int(size * 0.03), L["ny"] + net.height // 2 - 8), "net",
           fill=(120, 120, 120), font=font)

    label_font = load_bundled_font(max(13, size // 30), "sans-bold")
    for i, (sig, ok) in enumerate(prob["candidates"]):
        cube = render_cube(sig, prob["symbols"], prob["glyph_kind"],
                           prob["oriented"], L["cube_size"])
        cx, cy = _cube_xy(L, i, size)
        img.paste(cube, (cx, cy), cube)
        lab = str(i + 1)
        bb = d.textbbox((0, 0), lab, font=label_font)
        tw = bb[2] - bb[0]
        d.text((cx + L["cube_size"] / 2 - tw / 2, cy + L["cube_size"] + 1),
               lab, fill=(20, 20, 20), font=label_font)
    return img


def cube_readable(prob, size):
    """Every single-slot glyph turn and every candidate pair differ readably at `size`."""
    size = extra.resolve_size(size, prob["difficulty"])
    layout = _page_layout(prob, size)
    px = layout["cube_size"]
    if px < 78 or layout["cell_px"] < 28:
        return False
    images = []
    for sig, _ in prob["candidates"]:
        image = render_cube(sig, prob["symbols"], prob["glyph_kind"], True, px)
        images.append(image)
        for slot in ("top", "left", "right"):
            for turn in (1, 2, 3):
                probe = dict(sig)
                face, step = probe[slot]
                probe[slot] = (face, (step + turn) % 4)
                alternate = render_cube(
                    probe, prob["symbols"], prob["glyph_kind"], True, px
                )
                if not extra.difference(
                    image,
                    alternate,
                    contrast=48,
                    area=20,
                    core=3,
                    background=(250, 250, 252),
                )["readable"]:
                    return False
    return all(
        extra.difference(a, b, contrast=48, area=20, core=3, background=(250, 250, 252))[
            "readable"
        ]
        for a, b in combinations(images, 2)
    )


# The question states every convention the answer depends on
# (printed side outward, rotations only / no mirror images, symbol orientation
# counts, numbering of the cubes, answer format).
# A numeric format example such as "e.g. 1,3,5" would sometimes be a real gold and
# often share numbers with one, so the format is illustrated with cube numbers no page has (at most 12 cubes, so 31/34/38 can
# never be a matching cube), and the drawing convention (cubes seen from above,
# showing the top front corner) is stated instead of assumed.

QUESTION = (
    'Fold the top net into a cube, drawn side out. Which numbered cubes, each showing its three '
    'top-front-corner faces, are this cube after some rotation, with faces, layout and symbol '
    'orientations all matching (mirror images do not)? Answer the numbers ascending, '
    'comma-separated, no spaces.'
)


def make_cot(prob):
    n_good = len(prob["consistent_idx"])
    kind = ("orientation-sensitive symbols, so a cube matches only if the three "
            "visible faces are mutually adjacent AND each symbol's rotation agrees"
            if prob["oriented"] else
            "rotation-invariant colours, so a cube matches if its three visible "
            "faces are mutually adjacent with the correct cyclic arrangement")
    return (f"Folding the net gives a cube with fixed face adjacencies and "
            f"opposite-face pairs. Each candidate shows three faces meeting at a "
            f"corner. Using {kind}, a candidate is consistent iff its view is one "
            f"of the cube's legal corner views. Checking each candidate against all "
            f"24 rotations of the folded cube, {n_good} of them match: "
            f"{prob['answer']}.")


def generate_one(seed, idx, size, difficulty=None, *, difficulty_weights=(3, 4, 3)):
    requested_difficulty = difficulty
    weights = difficulty_weights
    diffs = ["easy", "medium", "hard"]
    rng = derive_rng(seed, idx)
    prob = None
    for _ in range(40):
        difficulty = requested_difficulty or rng.choices(diffs, weights=weights, k=1)[0]
        prob = build_problem(rng, difficulty)
        if prob is not None and (not DIFFICULTY[difficulty].get("readable")
                                 or cube_readable(prob, size)):
            break
        prob = None
    if prob is None:
        raise RuntimeError(f"could not build a cube-unfold problem for idx={idx}")
    img = render_page(prob, size)
    img_rel = f"images/cube_unfold_{idx:06d}.png"

    cand_meta = []
    for i, (sig, ok) in enumerate(prob["candidates"]):
        cand_meta.append({
            "index": i + 1,
            "consistent": bool(ok),
            "view": {s: list(sig[s]) for s in ("top", "right", "left")},
        })
    cfg = DIFFICULTY[prob["difficulty"]]
    rec = {
        "id": f"cube_unfold_{idx:06d}",
        "task_type": "spatial_perception",
        "subtype": "3d_cube_unfold",
        "level": "high",
        "difficulty": prob["difficulty"],
        "image": img_rel,
        "question": QUESTION,
        "ans_type": "blank",
        "options": [],
        "answer": prob["answer"],
        "cot": make_cot(prob),
        "meta": {
            "net_cells": [list(p) for p in prob["cells"]],
            "folded_faces": {str(fid): {"normal": list(n), "up": list(u)}
                             for fid, (n, u) in prob["folded"].items()},
            "oriented": prob["oriented"],
            "glyph_kind": prob["glyph_kind"],
            "large_glyphs": bool(cfg.get("large")),
            "readability_checked": bool(cfg.get("readable")),
            "rotation_only": bool(cfg.get("rotation_only")),
            "monochrome": bool(cfg.get("monochrome")),
            "resolution": size,
            "palette": prob["palette"],
            "face_colors": [list(s["color"]) for s in prob["symbols"]],
            "face_glyphs": [s.get("glyph", "dot") for s in prob["symbols"]],
            "generator_version": GENERATOR_VERSION,
            "slot_normals": SLOT_NORMAL,
            "slot_ref_up": {slot: REF_UP[n] for slot, n in SLOT_NORMAL.items()},
            "n_candidates": len(prob["candidates"]),
            "consistent_idx": prob["consistent_idx"],
            "candidates": cand_meta,
            "geometry": page_geometry(prob, size),
        },
    }
    return rec, img
