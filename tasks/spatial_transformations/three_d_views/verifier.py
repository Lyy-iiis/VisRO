"""Independent verifier for three_d_views.

Nothing here imports the generator (generator/generate.py, extra_difficulty.py) or
the shared renderer (shared/drawing/isometric.py).  The verifier has its own isometric
renderer, written from the documented 2:1 geometry (screen x = (i - j) * hw, y = (i + j) * hh -
k * ch, painter order by i + j), and uses it three ways:

1. Image binding.  The whole PNG must equal the verifier's drawing of meta pixel for pixel: the
   figure re-rendered from meta.heightmap, the arrow and FRONT/RIGHT word redrawn from
   meta.layout (which must follow the layout rule), the option panels of meta.option_grids, the
   letters, the slot outlines and the divider, drawn in the generator's order.  Overlays may not
   touch figure content.  The option panels and letters are also decoded from pixels (cell
   interiors must be exactly the fill colour or exactly white) for specific error messages.
2. Gold.  Camera conventions are derived, not copied: the screen images of the world axes plus
   the fact that the renderer shows the top, +i and +j faces fix a physical 3D frame; the camera
   right vector of each view is (view direction) x (up), with up = +k for front/side and up = -j
   ("the top of each option is the back, opposite FRONT") for top.  The asked view is decoded from
   the arrow pixels (largest arrow-blue component, principal axis, head end) and must match the
   question; the FRONT label must sit on the +j side (closest to +j of the four floor sides).
   The projection is then max-over-depth bars (front/side) or the footprint (top), laid out along
   the derived right/down axes.
3. Uniqueness, under the stacking rule (cubes stand on the drawn floor grid, each on a grid square
   or on the cube directly below it, so every candidate scene is a heightmap of solid columns on
   the drawn (N+1) x (M+1) grid) and the panel convention (the grid may extend past the figure, so
   options are matched by shape).
   a. Minimal witnesses (fast, gives the diagnostics): for every distractor, every minimal heightmap
      on the base whose view IS that distractor is rendered with the figure's own camera and must
      differ from the figure by a robust patch (contrast >= 32, area >= 48 px, eroded core >= 12 px;
      the repo's difference3d standard) outside the overlay.  A single-cell scan (every cell, every
      height 0..tier hmax) finds edits the figure cannot show; an invisible single-cell edit that
      changes the asked view at all is a problem, and each one is added to every witness.
   b. Exhaustive search (exhaustive_refutation): depth-first over every heightmap on the drawn grid,
      heights unbounded (a per-cell bound h* above which the column covers a robust patch of white
      background is derived from the pixels), keeping only scenes whose view has the shape of some
      distractor placed anywhere in the grid, pruned only by robust differences on pixels that no
      remaining column can change.  Any surviving leaf is a counterexample.  A node budget turns an
      unfinished search into a reject.
   Options that are translates of one another inside the frame are refused outright.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import functools
import itertools
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT_BOLD = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')

CORRECTNESS = (
    "Re-renders meta.heightmap with the verifier's own isometric renderer and redraws the whole image from meta "
    "(figure, arrow, FRONT/RIGHT word, option panels, letters, slot outlines, divider), requiring every pixel to "
    "match; decodes the asked view from the arrow pixels and the label, derives each camera's right/up axes from "
    "the rendered axis geometry, recomputes the orthographic projection, decodes all option panels and letters "
    "from pixels, requires exactly one panel (the answer letter) to equal the projection, and checks the tier "
    "(2^k corners of k disjoint edits of the tier's size, easy cube count, xhard/xxhard terraces, staircase "
    "options, top fill, view cycle) and that the question carries no format example.")
UNIQUENESS = (
    "Exhaustive under the question's stated rules: every scene of solid columns of any height on the drawn floor "
    "grid (base plus front padding strip) whose view has the shape of a distractor anywhere in the grid is "
    "proved to render with a robust visible difference (contrast >= 32, >= 48 px, core >= 12 px) from the "
    "figure under the same camera (depth-first search with sound pixel-finality pruning; a per-cell height "
    "bound above which a column shows on white background); no single-cell change that alters the asked view "
    "is invisible; no option is a translate of another.")

# Tier rows (must agree with the adapter's TIER_DEFINITIONS).  k = independent edits (2^k options);
# edit = size of each edit (top: footprint cells; front/side: levels one bar moves); cubes = easy's
# cube range per view class; terrace = xhard/xxhard scenes are descending terraces with staircase options;
# top_fill = footprint fill range of every top-view option.
TIERS = {
    # easy: top views on a 4x4, 3x5 or 4x5 base with 4-14 cubes, front/side views on 5x5 with 8-14
    # cubes (3-4 x 4 and 4-14 cubes for every view); 4x5 / 5x5 figures are drawn with a 38 / 30 px margin
    "easy":   dict(size=448, n=4, rows=(3, 5), cols=(4, 5), hmax=2, count_word="four", k=2, edit=(2, 3),
                   cubes={"top": (4, 14), "lateral": (8, 14)},
                   bases={"top": ((4, 4), (3, 5), (4, 5)), "lateral": ((5, 5),)},
                   margins={(4, 5): 38, (5, 5): 30}),
    "medium": dict(size=448, n=4, rows=(3, 3), cols=(3, 4), hmax=3, count_word="four", k=2, edit=(1, 2)),
    "hard":   dict(size=448, n=4, rows=(3, 4), cols=(3, 4), hmax=4, count_word="four", k=2, edit=(1, 1)),
    "xhard":  dict(size=588, n=4, rows=(4, 4), cols=(5, 5), hmax=4, count_word="four", k=2, edit=(1, 1),
                   terrace=True, top_fill=(0.25, 0.90)),
    "xxhard": dict(size=588, n=8, rows=(5, 5), cols=(5, 5), hmax=5, count_word="eight", k=3, edit=(1, 1),
                   terrace=True, top_fill=(0.25, 0.90)),
}
EXTRA_VIEW_CYCLE = ("front", "top", "side")     # xhard/xxhard: view = cycle[index % 3]
LETTERS = "ABCDEFGH"
VIEWS = ("front", "side", "top")

# Robust visual difference (the repo's 3D standard, extra_difficulty.difference3d).
CONTRAST, AREA, CORE = 32, 48, 12
MIN_SURFACE = 8          # robust interior px every column top / empty floor cell must keep
MIN_PANEL_CELL = 10      # px per option square
FIGURE_FRACTION = 0.58
SS = 2                   # renderer supersampling

WHITE = (255, 255, 255)
FLOOR_RGB = (200, 205, 212)
EDGE_RGB = (20, 20, 20)
ARROW_RGB = (40, 110, 220)
PANEL_FILL = (244, 120, 70)
SLOT_RGB = (225, 225, 230)
DIVIDER_RGB = (230, 230, 234)
BASE_HW, BASE_HH, BASE_CH = 34, 17, 40   # iso metrics before fitting; ss=2 multiplies them


def expected_question(view: str, n: int) -> str:
    letters = f"A-{LETTERS[n - 1]}"
    if view == "top":
        return (f"Which option {letters} is the top view of this cube figure, looking straight down? Filled squares "
                "are where the figure is seen (the grid may extend past it); the top of each option is the back, "
                "opposite FRONT. Answer with the letter.")
    name = "front" if view == "front" else "right-side"
    return (f"Which option {letters} is the {name} view of this cube figure, in the direction of the arrow? Filled "
            "squares are where the figure is seen (the grid may extend past it); the bottom row is floor level. "
            "Answer with the letter.")


_EXAMPLE = re.compile(r"(?:e\.g\.|for example|for instance|such as|like)\s*[:,]?\s*([^.;)?]*)", re.I)


def format_examples(question: str) -> list[str]:
    """Answer-format examples offered in the question text (after e.g./for example/such as/like, or
    quoted).  The templates carry none: the choice range "(A-D)" lists the valid answers."""
    found = [m.group(1).strip().strip('"\'') for m in _EXAMPLE.finditer(question)] + re.findall(r'"([^"]+)"', question)
    return list(dict.fromkeys(x for x in found if x))


def example_overlaps(question: str, answer: str) -> list[str]:
    """Examples that equal the gold or share a component (letter/word) with it."""
    out = []
    for ex in format_examples(question):
        parts = set(re.findall(r"[A-Za-z0-9]+", ex))
        if ex == answer or answer in parts:
            out.append(ex)
    return out


# --------------------------------------------------------------------------- #
# Own isometric renderer
# --------------------------------------------------------------------------- #
def _pt(i, j, k, cam):
    ox, oy, hw, hh, ch = cam
    return (ox + (i - j) * hw, oy + (i + j) * hh - k * ch)


def _extent(H, hw, hh, ch):
    N, M = H.shape
    pts = [(0, 0, 0), (M + 1, 0, 0), (0, N + 1, 0), (M + 1, N + 1, 0)]
    for j in range(N):
        for i in range(M):
            h = int(H[j, i])
            if h > 0:
                for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    pts += [(i + di, j + dj, h), (i + di, j + dj, 0)]
    xs = [(i - j) * hw for i, j, _ in pts]
    ys = [(i + j) * hh - k * ch for i, j, k in pts]
    return min(xs), min(ys), max(xs), max(ys)


def fit_camera(H, px, margin=46, ss=SS):
    """Scale the stack (floor drawn one cell larger) to fit px - 2*margin and centre it."""
    hw, hh, ch = BASE_HW * ss, BASE_HH * ss, BASE_CH * ss
    x0, y0, x1, y1 = _extent(H, hw, hh, ch)
    canvas = px * ss
    scale = min(1.0, (canvas - 2 * margin * ss) / max(x1 - x0, y1 - y0, 1))
    hw, hh, ch = max(2, int(hw * scale)), max(1, int(hh * scale)), max(2, int(ch * scale))
    x0, y0, x1, y1 = _extent(H, hw, hh, ch)
    return ((canvas - (x1 - x0)) / 2 - x0, (canvas - (y1 - y0)) / 2 - y0, hw, hh, ch)


def _faces(i, j, k, cam):
    P = lambda di, dj, dk: _pt(i + di, j + dj, k + dk, cam)  # noqa: E731
    top = [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]
    plus_j = [P(0, 1, 1), P(1, 1, 1), P(1, 1, 0), P(0, 1, 0)]    # face toward +j (screen left)
    plus_i = [P(1, 0, 1), P(1, 1, 1), P(1, 1, 0), P(1, 0, 0)]    # face toward +i (screen right)
    return top, plus_j, plus_i


def _painter(N, M):
    return sorted(((i, j) for j in range(N) for i in range(M)), key=lambda c: c[0] + c[1])


def render(H, px, cam, color, ss=SS) -> np.ndarray:
    return np.asarray(render_hi(H, px, cam, color, ss).resize((px, px), Image.Resampling.LANCZOS))


def render_hi(H, px, cam, color, ss=SS) -> Image.Image:
    """The supersampled drawing (before the LANCZOS downsample)."""
    H = np.asarray(H, dtype=int)
    N, M = H.shape
    img = Image.new("RGB", (px * ss, px * ss), WHITE)
    d = ImageDraw.Draw(img)
    for i in range(M + 2):
        d.line([_pt(i, 0, 0, cam), _pt(i, N + 1, 0, cam)], fill=FLOOR_RGB, width=ss)
    for j in range(N + 2):
        d.line([_pt(0, j, 0, cam), _pt(M + 1, j, 0, cam)], fill=FLOOR_RGB, width=ss)
    shades = [tuple(max(0, min(255, int(c * f))) for c in color) for f in (1.0, 0.74, 0.88)]
    for i, j in _painter(N, M):
        for k in range(int(H[j, i])):
            faces = _faces(i, j, k, cam)
            for poly, fill in zip(faces, shades):
                d.polygon(poly, fill=fill, outline=EDGE_RGB)
            for poly in faces:
                d.line(poly + [poly[0]], fill=EDGE_RGB, width=ss)
    return img


def changed_window(H, W, px, cam, ss=SS, pad=4):
    """Output-pixel box that contains every pixel a change from H to W can alter.

    Painter's order means only pixels covered by the old or new cubes of a changed column can
    change; their outlines reach <= ss source px further and LANCZOS-3 at 2x reads <= 3 output px
    around a pixel, so the box is the union of those columns' screen extents plus a margin.
    """
    ox, oy, hw, hh, ch = cam
    js, is_ = np.nonzero(np.asarray(W) != np.asarray(H))
    if not len(js):
        return None
    xs, ys = [], []
    for j, i in zip(js.tolist(), is_.tolist()):
        top = max(int(H[j][i]), int(W[j][i]))
        xs += [ox + (i - j - 1) * hw, ox + (i - j + 1) * hw]
        ys += [oy + (i + j) * hh - top * ch, oy + (i + j + 2) * hh]
    x0 = max(0, int(np.floor((min(xs) - 2 * ss) / ss)) - pad)
    y0 = max(0, int(np.floor((min(ys) - 2 * ss) / ss)) - pad)
    x1 = min(px, int(np.ceil((max(xs) + 2 * ss) / ss)) + pad)
    y1 = min(px, int(np.ceil((max(ys) + 2 * ss) / ss)) + pad)
    return x0, y0, x1, y1


def surface_counts(H, px, cam, ss=SS) -> np.ndarray:
    """Robust interior px of each column top (occupied) or floor square (empty), occlusion included."""
    H = np.asarray(H, dtype=int)
    N, M = H.shape
    img = Image.new("I", (px * ss, px * ss), 0)
    d = ImageDraw.Draw(img)
    for j in range(N):
        for i in range(M):
            d.polygon([_pt(i, j, 0, cam), _pt(i + 1, j, 0, cam), _pt(i + 1, j + 1, 0, cam), _pt(i, j + 1, 0, cam)],
                      fill=1 + j * M + i)
    for i in range(M + 2):
        d.line([_pt(i, 0, 0, cam), _pt(i, N + 1, 0, cam)], fill=0, width=ss)
    for j in range(N + 2):
        d.line([_pt(0, j, 0, cam), _pt(M + 1, j, 0, cam)], fill=0, width=ss)
    for i, j in _painter(N, M):
        h = int(H[j, i])
        for k in range(h):
            faces = _faces(i, j, k, cam)
            for f, poly in enumerate(faces):
                d.polygon(poly, fill=(1 + j * M + i) if (f == 0 and k == h - 1) else 0, outline=0)
            for poly in faces:
                d.line(poly + [poly[0]], fill=0, width=ss)
    arr = np.asarray(img).reshape(px, ss, px, ss).transpose(0, 2, 1, 3)
    lo, hi = arr.min(axis=(2, 3)), arr.max(axis=(2, 3))
    ids = np.where(lo == hi, lo, 0)
    pad = np.pad(ids, 1)
    core = ids.copy()
    for dy in range(3):
        for dx in range(3):
            core[pad[dy:dy + px, dx:dx + px] != ids] = 0
    return np.bincount(core.ravel(), minlength=N * M + 1)[1:].reshape(N, M)


def robust_difference(a: np.ndarray, b: np.ndarray, ignore=None) -> int:
    """Largest robust patch (contrast/area/core thresholds) where a and b differ; 0 = none."""
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    if ignore is not None:
        delta[ignore] = 0
    mask = delta >= CONTRAST
    if not mask.any():
        return 0
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    cores = np.bincount(lab[inner], minlength=n + 1)
    ok = (areas >= AREA) & (cores >= CORE)
    ok[0] = False
    return int(areas[ok].max()) if ok.any() else 0


# --------------------------------------------------------------------------- #
# Camera conventions, derived from the rendered axis geometry
# --------------------------------------------------------------------------- #
def world_frame(hw, hh, ch) -> np.ndarray:
    """Columns = physical 3D vectors of world steps +i, +j, +k (x right, y up, z toward viewer).

    Their screen components are fixed by the drawing; the depth components follow from
    orthogonality, with the sign fixed by the faces the renderer shows (top, +i and +j faces are
    visible, so all three axes point toward the viewer).
    """
    a = {"i": np.array([hw, -hh], float), "j": np.array([-hw, -hh], float), "k": np.array([0.0, ch])}
    p_ij, p_ik, p_jk = -a["i"] @ a["j"], -a["i"] @ a["k"], -a["j"] @ a["k"]
    if min(p_ij, p_ik, p_jk) <= 0:
        raise ValueError("camera metrics do not describe a view from above the front")
    z_i = np.sqrt(p_ij * p_ik / p_jk)
    z = {"i": z_i, "j": p_ij / z_i, "k": p_ik / z_i}
    return np.stack([np.append(a[c], z[c]) for c in "ijk"], axis=1)


def screen_dir(world, hw, hh, ch):
    """Image-space (y down) direction of a world direction (i, j, k)."""
    di, dj, dk = world
    return np.array([(di - dj) * hw, (di + dj) * hh - dk * ch], float)


VIEW_DIRECTION = {"front": (0, -1, 0), "side": (-1, 0, 0), "top": (0, 0, -1)}


def view_layout(view, E):
    """(right axis, right sign, down axis, down sign) of the view in world axes 'i', 'j', 'k'."""
    e = {c: E[:, n] for n, c in enumerate("ijk")}
    d = sum(s * e[c] for s, c in zip(VIEW_DIRECTION[view], "ijk"))
    up = -e["j"] if view == "top" else e["k"]
    right = np.cross(d, up)
    def axis(v):
        w = np.linalg.solve(E, v)
        n = int(np.argmax(np.abs(w)))
        if abs(w[n]) < 0.9 * np.linalg.norm(w):
            raise ValueError("view axis is not aligned with a world axis")
        return "ijk"[n], int(np.sign(w[n]))
    return axis(right) + axis(-up)


def project(H, view, frame_height, E) -> np.ndarray:
    H = np.asarray(H, dtype=int)
    r_axis, r_sign, d_axis, d_sign = view_layout(view, E)
    arr_axis = {"j": 0, "i": 1}
    if view == "top":
        assert {r_axis, d_axis} == {"i", "j"}
        g = (H > 0).astype(int)
        if arr_axis[r_axis] == 0:      # columns must run along the right axis
            g = g.T
        if r_sign < 0:
            g = g[:, ::-1]
        if d_sign < 0:
            g = g[::-1, :]
        return g
    assert d_axis == "k" and d_sign == -1 and r_axis in arr_axis
    depth = 1 - arr_axis[r_axis]
    bars = H.max(axis=depth)
    if r_sign < 0:
        bars = bars[::-1]
    levels = np.arange(frame_height, 0, -1)[:, None]   # row 0 = highest level
    return (bars[None, :] >= levels).astype(int)


def lines_of(view, shape, E):
    """For lateral views: the world cells (j, i) behind each displayed column, in display order."""
    N, M = shape
    r_axis, r_sign, _, _ = view_layout(view, E)
    if r_axis == "i":
        cols = [[(j, i) for j in range(N)] for i in range(M)]
    else:
        cols = [[(j, i) for i in range(M)] for j in range(N)]
    return cols[::-1] if r_sign < 0 else cols


def cell_of_top(view, shape, E):
    """For the top view: world cell (j, i) shown at display (row, col)."""
    N, M = shape
    r_axis, r_sign, d_axis, d_sign = view_layout(view, E)
    out = {}
    for j in range(N):
        for i in range(M):
            w = {"i": i, "j": j}
            n = {"i": M, "j": N}
            col = w[r_axis] if r_sign > 0 else n[r_axis] - 1 - w[r_axis]
            row = w[d_axis] if d_sign > 0 else n[d_axis] - 1 - w[d_axis]
            out[(row, col)] = (j, i)
    return out


# --------------------------------------------------------------------------- #
# Uniqueness: minimal 3D witnesses of each distractor
# --------------------------------------------------------------------------- #
def crop_ink(g):
    g = np.asarray(g)
    rows, cols = np.flatnonzero(g.any(axis=1)), np.flatnonzero(g.any(axis=0))
    if not len(rows):
        return g[:0, :0]
    return g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def is_translate(truth, grid) -> bool:
    if np.array_equal(truth, grid):
        return False
    a, b = crop_ink(truth), crop_ink(grid)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def minimal_witnesses(H, view, truth, D, wmax, E):
    """Heightmaps whose view equals D and that change H as little as possible, line by line."""
    H = np.asarray(H, dtype=int)
    per_change = []
    if view == "top":
        where = cell_of_top(view, H.shape, E)
        for r, c in zip(*np.nonzero(truth != D)):
            cell = where[(int(r), int(c))]
            if D[r, c] == 0:
                per_change.append([[(cell, 0)]])
            else:
                per_change.append([[(cell, h)] for h in range(1, wmax + 1)])
    else:
        lines = lines_of(view, H.shape, E)
        tb, db = truth.sum(axis=0), D.sum(axis=0)
        for c in np.flatnonzero(tb != db):
            v = int(db[c])
            if v < tb[c]:
                per_change.append([[(cell, min(int(H[cell]), v)) for cell in lines[c]]])
            else:
                per_change.append([[(cell, v)] for cell in lines[c]])
    for combo in itertools.product(*per_change):
        W = H.copy()
        for edits in combo:
            for cell, v in edits:
                W[cell] = v
        yield W


def analyse_uniqueness(H, view, truth, options, answer_index, wmax, cam, px, color, ignore, E):
    """Returns (problems, stats).  stats is diagnostic only."""
    H = np.asarray(H, dtype=int)
    F = options[0].shape[0]          # lateral frame height (rows of every option)
    ref = render(H, px, cam, color)

    def same_view(W, G):
        """W's view is the grid G (a lateral view must also fit the option frame)."""
        if view == "top":
            return np.array_equal(project(W, view, F, E), G)
        return int(W.max()) <= F and np.array_equal(project(W, view, F, E), G)

    @functools.lru_cache(maxsize=None)
    def visible(key):
        W = np.frombuffer(key, dtype=np.int64).reshape(H.shape)
        box = changed_window(H, W, px, cam)
        if box is None:
            return 0
        x0, y0, x1, y1 = box
        out = np.asarray(render_hi(W, px, cam, color).resize(
            (x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=(x0 * SS, y0 * SS, x1 * SS, y1 * SS)))
        return robust_difference(ref[y0:y1, x0:x1], out, ignore[y0:y1, x0:x1])

    def diff_of(W):
        return visible(np.ascontiguousarray(W, dtype=np.int64).tobytes())

    problems = []
    invisible = []
    for j in range(H.shape[0]):
        for i in range(H.shape[1]):
            for v in range(wmax + 1):
                if v == H[j, i]:
                    continue
                W = H.copy()
                W[j, i] = v
                if diff_of(W) == 0:
                    invisible.append(((j, i), v))
    hidden_view_changes = 0
    for cell, v in invisible:
        W = H.copy()
        W[cell] = v
        if same_view(W, truth):
            continue
        hidden_view_changes += 1
        if hidden_view_changes == 1:
            problems.append(f"cell (j={cell[0]}, i={cell[1]}) is not robustly shown: height {v} instead of "
                            f"{int(H[cell])} changes the asked view but renders without a robust difference")
    min_margin = None
    for k, D in enumerate(options):
        if k == answer_index:
            continue
        if is_translate(truth, D):
            problems.append(f"option {LETTERS[k]} is the truth translated inside the frame")
            continue
        n_w = 0
        for W in minimal_witnesses(H, view, truth, D, wmax, E):
            n_w += 1
            if not same_view(W, D):
                problems.append(f"internal: witness for {LETTERS[k]} does not project to it")
                break
            margin = diff_of(W)
            if margin == 0:
                problems.append(f"option {LETTERS[k]} not robustly refuted: heightmap {W.tolist()} has that "
                                f"view and renders without a robust difference")
                break
            min_margin = margin if min_margin is None else min(min_margin, margin)
            bad = None
            for cell, v in invisible:
                if W[cell] != H[cell]:
                    continue
                W2 = W.copy()
                W2[cell] = v
                if same_view(W2, D) and diff_of(W2) == 0:
                    bad = W2
                    break
            if bad is not None:
                problems.append(f"option {LETTERS[k]} not robustly refuted: heightmap {bad.tolist()} has that "
                                f"view and renders without a robust difference")
                break
        if n_w == 0:
            problems.append(f"internal: no witness enumerated for option {LETTERS[k]}")
    stats = dict(invisible_single_edits=len(invisible), hidden_view_changes=hidden_view_changes,
                 min_refutation_px=min_margin)
    return problems, stats


# --------------------------------------------------------------------------- #
# Uniqueness, exhaustive: every scene the stacking rule allows
# --------------------------------------------------------------------------- #
# The stacking rule allows any solid columns on the DRAWN floor grid: the N x M base plus the front
# padding strip (row j = N, column i = M), of any height.  The question says the option grid may
# extend past the figure, so an option is matched by its shape: a scene whose view is the shape of a
# distractor placed anywhere in the grid makes that distractor a valid answer.  The search below
# visits every such scene whose view has the shape of some distractor and proves it renders with a
# robust difference from the figure.
#
# Depth-first over the cells of the drawn grid in reverse painter order (the column drawn last
# first).  A hi-res pixel is final once it is covered by an assigned column (assigned columns are
# drawn after every unassigned one) or lies outside every unassigned column's largest silhouette; an
# output pixel is final once its whole LANCZOS support is.  Differences on final output pixels are
# shared by every completion of the branch, and a robust patch of a subset of a mask is a robust
# patch of the mask (area and eroded core only grow), so a robust patch there refutes the whole
# branch.  Leaves are re-rendered in full and tested with robust_difference.
# Unbounded heights: for each cell the search finds the least height h* at which the column covers,
# with its own faces, a robust patch of output pixels that are pure white in the figure (pixels whose
# whole LANCZOS support lies inside the column of height h*; a taller column, or any column drawn
# after it, still covers them with face or edge colours, green <= 120, so they differ by >= 100).
# Every height >= h* is therefore refuted, and the search tries 0 .. h*-1.
EXH_NODE_LIMIT = 40000
EXH_HEIGHT_LIMIT = 40
_SENT = (1, 2, 3)


def _column_patch(i, j, h, cam, color, size_hi):
    """(y0, x0, coverage, rgb) of one column drawn exactly as render_hi draws it, cut to its box.
    The camera origin is a half-integer, so shifting it by whole pixels is exact."""
    ox, oy, hw, hh, ch = cam
    xs = (ox + (i - j - 1) * hw, ox + (i - j + 1) * hw)
    ys = (oy + (i + j) * hh - h * ch, oy + (i + j + 2) * hh)
    x0, y0 = max(0, int(np.floor(xs[0])) - 4), max(0, int(np.floor(ys[0])) - 4)
    x1, y1 = min(size_hi, int(np.ceil(xs[1])) + 5), min(size_hi, int(np.ceil(ys[1])) + 5)
    if x1 <= x0 or y1 <= y0:
        return y0, x0, np.zeros((0, 0), bool), np.zeros((0, 0, 3), np.uint8)
    sub = (ox - x0, oy - y0, hw, hh, ch)
    im = Image.new("RGB", (x1 - x0, y1 - y0), _SENT)
    d = ImageDraw.Draw(im)
    shades = [tuple(max(0, min(255, int(c * f))) for c in color) for f in (1.0, 0.74, 0.88)]
    for k in range(h):
        faces = _faces(i, j, k, sub)
        for poly, fill in zip(faces, shades):
            d.polygon(poly, fill=fill, outline=EDGE_RGB)
        for poly in faces:
            d.line(poly + [poly[0]], fill=EDGE_RGB, width=SS)
    a = np.asarray(im)
    return y0, x0, np.any(a != np.array(_SENT, np.uint8), axis=2), a


def _robust_mask(mask) -> bool:
    if mask.sum() < AREA:
        return False
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    sub = np.pad(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], 1)
    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    if areas[1:].max() < AREA:
        return False
    cores = np.bincount(lab[ndimage.binary_erosion(sub, structure=np.ones((3, 3)))], minlength=n + 1)
    ok = (areas >= AREA) & (cores >= CORE)
    ok[0] = False
    return bool(ok.any())


def _support_inside(cov_full, px):
    """Output pixels whose hi-res support [2y-8, 2y+9] x [2x-8, 2x+9] lies inside cov_full."""
    c = cov_full
    blk = c[0::2, 0::2] & c[1::2, 0::2] & c[0::2, 1::2] & c[1::2, 1::2]     # SS == 2
    return ndimage.minimum_filter(blk.view(np.uint8), size=9, mode="constant", cval=0).astype(bool)


def shape_targets(view, options, answer_index, ext_shape, E):
    """Every placement, on the drawn grid, of the shape of every distractor.
    top: list of (option index, {cell: 0/1}); front/side: list of (option index, bars per line)."""
    Ne, Me = ext_shape
    out = []
    for k, D in enumerate(options):
        if k == answer_index:
            continue
        if view == "top":
            C = crop_ink(D)
            where = cell_of_top(view, ext_shape, E)
            R = max(r for r, _ in where) + 1
            Cc = max(c for _, c in where) + 1
            for r0 in range(R - C.shape[0] + 1):
                for c0 in range(Cc - C.shape[1] + 1):
                    G = np.zeros((R, Cc), int)
                    G[r0:r0 + C.shape[0], c0:c0 + C.shape[1]] = C
                    out.append((k, {where[(r, c)]: int(G[r, c]) for (r, c) in where}))
        else:
            b = D.sum(axis=0)
            nz = np.flatnonzero(b)
            seq = b[nz[0]:nz[-1] + 1]
            L = len(lines_of(view, ext_shape, E))
            for o in range(L - len(seq) + 1):
                bars = np.zeros(L, int)
                bars[o:o + len(seq)] = seq
                out.append((k, bars))
    return out


def exhaustive_refutation(H, view, options, answer_index, cam, px, color, ignore, E, ref,
                          node_limit=EXH_NODE_LIMIT, collect=False, height_cap=None):
    """(counterexample, stats).  counterexample = (option index, scene on the drawn grid) of a scene
    the stacking rule allows, whose view has a distractor's shape and which renders without a robust
    difference; None when every such scene is refuted.  stats['aborted'] marks an exhausted budget.
    Audit hooks: collect=True returns the list of ALL such scenes; height_cap limits the heights tried
    (for a brute-force cross-check over the same finite space)."""
    H = np.asarray(H, dtype=int)
    N, M = H.shape
    Ne, Me = N + 1, M + 1
    hi = px * SS
    Hx = np.zeros((Ne, Me), int)
    Hx[:N, :M] = H
    order = _painter(Ne, Me)[::-1]
    stats = dict(nodes=0, aborted=False)
    cache = {}

    def col(i, j, h):
        if (i, j, h) not in cache:
            cache[(i, j, h)] = _column_patch(i, j, h, cam, color, hi)
        return cache[(i, j, h)]

    def full_cov(i, j, h):
        y0, x0, cov, _ = col(i, j, h)
        m = np.zeros((hi, hi), bool)
        m[y0:y0 + cov.shape[0], x0:x0 + cov.shape[1]] = cov
        return m

    white = np.all(ref == 255, axis=2) & ~ignore
    targets = shape_targets(view, options, answer_index, (Ne, Me), E)
    if view == "top":
        hstar = {}
        for (i, j) in order:
            for h in range(1, EXH_HEIGHT_LIMIT + 1):
                if _robust_mask(_support_inside(full_cov(i, j, h), px) & white):
                    hstar[(i, j)] = h
                    break
            else:
                return None, dict(stats, aborted=True, reason=f"no height bound at cell ({j},{i})")
        line_of = None
    else:
        lines = lines_of(view, (Ne, Me), E)
        line_of = {cell: n for n, line in enumerate(lines) for cell in line}
        last_of_line = {}
        for (i, j) in order:
            last_of_line[line_of[(j, i)]] = (i, j)
        top_bar = max(int(b.max()) for _, b in targets)
        hstar = {c: top_bar + 1 for c in order}
    # floor drawn for the base, as in the figure
    floor = Image.new("RGB", (hi, hi), WHITE)
    d = ImageDraw.Draw(floor)
    for i in range(M + 2):
        d.line([_pt(i, 0, 0, cam), _pt(i, N + 1, 0, cam)], fill=FLOOR_RGB, width=SS)
    for j in range(N + 2):
        d.line([_pt(0, j, 0, cam), _pt(M + 1, j, 0, cam)], fill=FLOOR_RGB, width=SS)
    floor = np.asarray(floor)

    def compose(Wx):
        canvas, covall = floor.copy(), np.zeros((hi, hi), bool)
        for (i, j) in order:
            h = int(Wx[j, i])
            if h:
                y0, x0, cov, rgb = col(i, j, h)
                sl = (slice(y0, y0 + cov.shape[0]), slice(x0, x0 + cov.shape[1]))
                put = cov & ~covall[sl]
                canvas[sl][put] = rgb[put]
                covall[sl] |= cov
        return canvas

    if not np.array_equal(compose(Hx), np.asarray(render_hi(H, px, cam, color))):
        return None, dict(stats, aborted=True, reason="column composition differs from the renderer")
    pend = np.zeros((hi, hi), np.int16)
    sil, wins = {}, {}
    for (i, j) in order:
        y0, x0, cov, _ = col(i, j, hstar[(i, j)] - 1) if hstar[(i, j)] > 1 else col(i, j, 1)
        m = np.zeros((hi, hi), bool)
        m[y0:y0 + cov.shape[0], x0:x0 + cov.shape[1]] = cov
        m = ndimage.binary_dilation(m, iterations=2)
        if not m.any():
            m[0, 0] = True
        ys, xs = np.nonzero(m)
        yy0, yy1, xx0, xx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sil[(i, j)] = (yy0, xx0, yy1, xx1, m[yy0:yy1, xx0:xx1])
        pend[yy0:yy1, xx0:xx1] += m[yy0:yy1, xx0:xx1]
        wins[(i, j)] = (max(0, (yy0 - 10) // SS), max(0, (xx0 - 10) // SS),
                        min(px, (yy1 + 10) // SS + 1), min(px, (xx1 + 10) // SS + 1))
    canvas = floor.copy()
    covall = np.zeros((hi, hi), bool)
    # delta: robustly-contrasting output pixels of the current canvas (final or not); dmask: the final
    # ones outside the overlay.  Only pixels a column covers change the canvas, so only their LANCZOS
    # neighbourhood is resampled when a column is placed.
    delta = np.abs(np.asarray(Image.fromarray(canvas).resize((px, px), Image.Resampling.LANCZOS)).astype(np.int16)
                   - ref.astype(np.int16)).max(axis=2) >= CONTRAST
    dmask = np.zeros((px, px), bool)
    Wx = Hx.copy()
    ref16 = ref.astype(np.int16)
    found = []

    def final_out(win):
        oy0, ox0, oy1, ox1 = win
        by0, bx0, by1, bx1 = max(0, oy0 - 4), max(0, ox0 - 4), min(px, oy1 + 5), min(px, ox1 + 5)
        fin = covall[SS * by0:SS * by1, SS * bx0:SS * bx1] | (pend[SS * by0:SS * by1, SS * bx0:SS * bx1] == 0)
        blk = ~(fin[0::2, 0::2] & fin[1::2, 0::2] & fin[0::2, 1::2] & fin[1::2, 1::2])   # SS == 2
        nf = ndimage.maximum_filter(blk.view(np.uint8), size=9, mode="constant", cval=0)
        return nf[oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0] == 0

    def resample(y0, x0, y1, x1):
        """Recompute delta on the output pixels whose LANCZOS support meets hi-res rows y0..y1, cols x0..x1."""
        oy0, ox0 = max(0, (y0 - 9) // SS), max(0, (x0 - 9) // SS)
        oy1, ox1 = min(px, (y1 + 9) // SS + 1), min(px, (x1 + 9) // SS + 1)
        cy0, cx0 = max(0, oy0 * SS - 12), max(0, ox0 * SS - 12)
        cy1, cx1 = min(hi, oy1 * SS + 12), min(hi, ox1 * SS + 12)
        im = Image.fromarray(np.ascontiguousarray(canvas[cy0:cy1, cx0:cx1]))
        out = np.asarray(im.resize((ox1 - ox0, oy1 - oy0), Image.Resampling.LANCZOS,
                                   box=(ox0 * SS - cx0, oy0 * SS - cy0, ox1 * SS - cx0, oy1 * SS - cy0)))
        delta[oy0:oy1, ox0:ox1] = np.abs(out.astype(np.int16) - ref16[oy0:oy1, ox0:ox1]).max(axis=2) >= CONTRAST

    def robust_near(win):
        """Robust patch in dmask around the window (a robust patch of part of a mask is one of the mask;
        any new one touches the window, and a margin keeps nearly all of it)."""
        oy0, ox0, oy1, ox1 = win
        return _robust_mask(dmask[max(0, oy0 - 24):oy1 + 24, max(0, ox0 - 24):ox1 + 24])

    def allowed(i, j, K):
        hmax_c = hstar[(i, j)] - 1 if height_cap is None else min(hstar[(i, j)] - 1, height_cap)
        if view == "top":
            want = {targets[t][1][(j, i)] for t in K}
            hs = ([0] if 0 in want else []) + (list(range(1, hmax_c + 1)) if 1 in want else [])
        else:
            ln = line_of[(j, i)]
            hs = list(range(0, min(hmax_c, max(int(targets[t][1][ln]) for t in K)) + 1))
        h0 = int(Hx[j, i])
        return ([h0] if h0 in hs else []) + [h for h in hs if h != h0]

    def narrow(i, j, h, K):
        if view == "top":
            return [t for t in K if targets[t][1][(j, i)] == (h > 0)]
        ln = line_of[(j, i)]
        K2 = [t for t in K if h <= targets[t][1][ln]]
        if last_of_line[ln] == (i, j):
            mx = max(int(Wx[c]) for c in lines[ln])
            K2 = [t for t in K2 if mx == targets[t][1][ln]]
        return K2

    def dfs(depth, K):
        stats["nodes"] += 1
        if stats["nodes"] > node_limit:
            stats["aborted"] = True
            return
        if depth == len(order):
            out = np.asarray(Image.fromarray(compose(Wx)).resize((px, px), Image.Resampling.LANCZOS))
            if robust_difference(ref, out, ignore) == 0:
                found.append((sorted({targets[t][0] for t in K}), Wx.copy()) if collect
                             else (targets[K[0]][0], Wx.copy()))
            return
        i, j = order[depth]
        oy0, ox0, oy1, ox1 = win = wins[(i, j)]
        yy0, xx0, yy1, xx1, ms = sil[(i, j)]
        pend[yy0:yy1, xx0:xx1] -= ms
        save_c = canvas[yy0:yy1, xx0:xx1].copy()
        save_v = covall[yy0:yy1, xx0:xx1].copy()
        save_m = dmask[oy0:oy1, ox0:ox1].copy()
        save_d = delta[oy0:oy1, ox0:ox1].copy()
        ign = ignore[oy0:oy1, ox0:ox1]
        dirty = False
        for h in allowed(i, j, K):
            Wx[j, i] = h
            K2 = narrow(i, j, h, K)
            if not K2:
                continue
            if dirty:
                canvas[yy0:yy1, xx0:xx1] = save_c
                covall[yy0:yy1, xx0:xx1] = save_v
                delta[oy0:oy1, ox0:ox1] = save_d
                dirty = False
            if h:
                y0, x0, cov, rgb = col(i, j, h)
                sl = (slice(y0, y0 + cov.shape[0]), slice(x0, x0 + cov.shape[1]))
                put = cov & ~covall[sl]
                canvas[sl][put] = rgb[put]
                covall[sl] |= cov
                resample(y0, x0, y0 + cov.shape[0], x0 + cov.shape[1])
                dirty = True
            dmask[oy0:oy1, ox0:ox1] = delta[oy0:oy1, ox0:ox1] & final_out(win) & ~ign
            if robust_near(win):
                continue
            dfs(depth + 1, K2)
            if (found and not collect) or stats["aborted"]:
                break
        Wx[j, i] = Hx[j, i]
        canvas[yy0:yy1, xx0:xx1] = save_c
        covall[yy0:yy1, xx0:xx1] = save_v
        delta[oy0:oy1, ox0:ox1] = save_d
        dmask[oy0:oy1, ox0:ox1] = save_m
        pend[yy0:yy1, xx0:xx1] += ms

    dfs(0, list(range(len(targets))))
    stats["targets"] = len(targets)
    if collect:
        return found, stats
    return (found[0] if found else None), stats


# --------------------------------------------------------------------------- #
# Image decoding helpers
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _font(size):
    return ImageFont.truetype(str(FONT_BOLD), size)


def _text_ink(size_wh, xy, text, font_size, color):
    canvas = Image.new("RGB", size_wh, WHITE)
    ImageDraw.Draw(canvas).text(tuple(xy), text, fill=tuple(color), font=_font(font_size))
    arr = np.asarray(canvas)
    return arr, np.any(arr != 255, axis=2)


def decode_arrow(img: np.ndarray):
    """Largest arrow-blue component -> (tail, tip, unit direction, bbox) in image coords (y down)."""
    blue = np.all(img == np.array(ARROW_RGB, dtype=img.dtype), axis=2)
    lab, n = ndimage.label(blue, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    comp = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(comp)
    if len(xs) < 60:
        return None
    pts = np.stack([xs, ys], axis=1).astype(float)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    t = (pts - mean) @ axis
    s = (pts - mean) @ np.array([-axis[1], axis[0]])
    hi = np.ptp(s[t > t.max() - 12])
    lo = np.ptp(s[t < t.min() + 12])
    if hi == lo:
        return None
    if lo > hi:
        axis, t = -axis, -t
    tip = mean + axis * t.max()
    tail = mean + axis * t.min()
    return dict(tail=tail, tip=tip, dir=axis, mask=comp)


def _angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def draw_panel(g, cell, pad) -> np.ndarray:
    """An option panel as the question draws it: filled squares with dark outlines, empty squares
    with light outlines, row by row."""
    R, C = g.shape
    im = Image.new("RGB", (C * cell + 2 * pad, R * cell + 2 * pad), WHITE)
    d = ImageDraw.Draw(im)
    for r in range(R):
        for c in range(C):
            x0, y0 = pad + c * cell, pad + r * cell
            if g[r, c]:
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=PANEL_FILL, outline=(40, 40, 40), width=2)
            else:
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=WHITE, outline=(220, 220, 224), width=1)
    return np.asarray(im)


def decode_panel(img, box, cell, pad):
    """Binary grid from the cell interiors of one option panel, or an error string."""
    x0, y0, x1, y1 = box
    W, Hh = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    if cell < MIN_PANEL_CELL or W % cell or Hh % cell or W <= 0 or Hh <= 0:
        return f"panel box {box} is not a whole grid of {cell}px cells"
    R, C = Hh // cell, W // cell
    g = np.zeros((R, C), int)
    for r in range(R):
        for c in range(C):
            cx0, cy0 = x0 + pad + c * cell + 3, y0 + pad + r * cell + 3
            patch = img[cy0:cy0 + cell - 5, cx0:cx0 + cell - 5].reshape(-1, 3)
            if np.all(patch == PANEL_FILL):
                g[r, c] = 1
            elif not np.all(patch == 255):
                return f"panel cell ({r},{c}) in box {box} is neither filled nor empty"
    return g


# --------------------------------------------------------------------------- #
# Tier membership: options = 2^k corners of k disjoint edits; terraces
# --------------------------------------------------------------------------- #
def corner_edits(options, truth):
    """The k edit masks when the options are exactly the 2^k corners truth ^ (union of a subset of k
    pairwise disjoint non-empty masks); None otherwise."""
    X = [np.asarray(g, dtype=int) ^ truth for g in options]
    n = len(X)
    k = n.bit_length() - 1
    if n < 2 or (1 << k) != n:
        return None
    keys = {x.tobytes() for x in X}
    nz = [x for x in X if x.any()]
    atoms = [x for x in nz if not any(y.any() and not np.array_equal(y, x) and np.array_equal(y & x, y)
                                      and (x ^ y).tobytes() in keys for y in nz)]
    if len(atoms) != k or any((a & b).any() for a, b in itertools.combinations(atoms, 2)):
        return None
    corners = set()
    for bits in range(n):
        acc = np.zeros_like(truth)
        for i, a in enumerate(atoms):
            if bits >> i & 1:
                acc = acc | a
        corners.add(acc.tobytes())
    return atoms if corners == keys else None


def is_terrace(H) -> bool:
    H = np.asarray(H)
    return bool((np.diff(H, axis=0) <= 0).all() and (np.diff(H, axis=1) <= 0).all())


def staircase_option(g, view) -> bool:
    """xhard/xxhard option shapes: a top footprint that is an order ideal of the back-left corner, front
    bars non-increasing, side bars non-decreasing (the shapes every terrace has)."""
    g = np.asarray(g)
    if view == "top":
        return is_terrace(g)
    d = np.diff(g.sum(axis=0))
    return bool((d <= 0).all()) if view == "front" else bool((d >= 0).all())


def tier_problems(spec, view, H, options, truth, index=None) -> list[str]:
    P = []
    atoms = corner_edits(options, truth)
    if atoms is None or len(atoms) != spec["k"]:
        return [f"options are not the {len(options)} corners of {spec['k']} disjoint edits of the truth"]
    lo, hi = spec["edit"]
    for a in atoms:
        if view != "top" and np.count_nonzero(a.any(axis=0)) != 1:
            P.append("a front/side edit changes more than one bar")
        elif not lo <= int(a.sum()) <= hi:
            unit = "cells" if view == "top" else "levels"
            P.append(f"an edit changes {int(a.sum())} {unit}, outside the tier's {lo}-{hi}")
    if "cubes" in spec:
        lo_c, hi_c = spec["cubes"]["top" if view == "top" else "lateral"]
        if not lo_c <= int(H.sum()) <= hi_c:
            P.append(f"{int(H.sum())} cubes, outside the tier's {lo_c}-{hi_c} for a {view} view")
    if spec.get("terrace"):
        if not is_terrace(H):
            P.append("scene is not a descending terrace")
        if not all(staircase_option(g, view) for g in options):
            P.append("an option is not a staircase shape")
        if view == "top" and not all(spec["top_fill"][0] <= float(g.mean()) <= spec["top_fill"][1] for g in options):
            P.append("a top-view option's fill is outside the tier's range")
        if index is not None and view != EXTRA_VIEW_CYCLE[int(index) % 3]:
            P.append("view does not follow the tier's front/top/side cycle")
    return P


# --------------------------------------------------------------------------- #
# The image as the generator's layout draws it
# --------------------------------------------------------------------------- #
ARROW_WIDTH, ARROW_HEAD = 5, 13
EXTRA_LETTER = dict(font_size=16, color=(20, 20, 20), pad=4)
BASE_LETTER = dict(font_size=18, color=(30, 30, 30), pad=6)


def draw_overlay(d, lay):
    """Arrow (shaft + two head strokes at +-150 deg from the tip) and the FRONT/RIGHT word."""
    (ax, ay), (bx, by) = lay["arrow"]["segment"]
    d.line([(ax, ay), (bx, by)], fill=ARROW_RGB, width=ARROW_WIDTH)
    ang = math.atan2(by - ay, bx - ax)
    for da in (math.radians(150), math.radians(-150)):
        d.line([(bx, by), (bx + ARROW_HEAD * math.cos(ang + da), by + ARROW_HEAD * math.sin(ang + da))],
               fill=ARROW_RGB, width=ARROW_WIDTH)
    tl = lay["text_labels"][0]
    d.text(tuple(tl["xy"]), str(tl["text"]), fill=ARROW_RGB, font=_font(int(tl["font_size"])))


def expected_image(lay, labels, options, ref, size):
    """Every pixel of the question image, drawn from meta in the generator's order: figure, overlay,
    then the option zone (lower tiers: per option letter, panel, rounded slot outline, then the
    divider; xhard/xxhard: divider, then per option panel and letter)."""
    img = Image.new("RGB", (size, size), WHITE)
    fx, fy = (int(v) for v in lay["figure"]["offset"])
    img.paste(Image.fromarray(ref), (fx, fy))
    d = ImageDraw.Draw(img)
    draw_overlay(d, lay)
    pan = lay["panels"]
    cell, pad, zone_top = int(pan["cell"]), int(pan["pad"]), int(pan["zone_top"])
    divider = [(8, zone_top - 6), (size - 8, zone_top - 6)]
    slots = pan.get("slots") or []
    font = _font(int(pan["label_font_size"]))
    color = tuple(int(c) for c in pan["label_color"])
    if slots:
        for k, g in enumerate(options):
            d.text(tuple(pan["label_xy"][k]), labels[k], fill=color, font=font)
            img.paste(Image.fromarray(draw_panel(g, cell, pad)), tuple(int(v) for v in pan["boxes"][k][:2]))
            d.rounded_rectangle([int(v) for v in slots[k]], radius=8, outline=SLOT_RGB, width=2)
        d.line(divider, fill=DIVIDER_RGB, width=2)
    else:
        d.line(divider, fill=DIVIDER_RGB, width=2)
        for k, g in enumerate(options):
            img.paste(Image.fromarray(draw_panel(g, cell, pad)), tuple(int(v) for v in pan["boxes"][k][:2]))
            d.text(tuple(pan["label_xy"][k]), labels[k], fill=color, font=font)
    return np.asarray(img)


def overlay_mask(lay, size):
    """Pixels the arrow, the word and the divider touch (drawn alone on white)."""
    img = Image.new("RGB", (size, size), WHITE)
    d = ImageDraw.Draw(img)
    draw_overlay(d, lay)
    zt = int(lay["panels"]["zone_top"])
    d.line([(8, zt - 6), (size - 8, zt - 6)], fill=DIVIDER_RGB, width=2)
    return np.any(np.asarray(img) != 255, axis=2)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def _as_grid(x):
    g = np.asarray(x)
    if g.ndim != 2 or g.size == 0 or not np.isin(g, (0, 1)).all():
        raise ValueError("not a non-empty binary grid")
    return g.astype(int)


def verify(record: dict, image) -> list[str]:
    try:
        problems, _ = check(record, image)
    except (KeyError, TypeError, IndexError) as exc:   # malformed meta/layout is a reject, not a crash
        return [f"malformed record: {exc!r}"]
    return problems


def check(record: dict, image, *, uniqueness: bool = True, surface_guard: bool = True, exhaustive: bool = True):
    """(problems, diagnostics).  Diagnostics feed the audit script, never the verdict.

    surface_guard=False skips the per-cell visibility count, exhaustive=False the exhaustive search
    (audits use them to show what each layer of the uniqueness proof catches on its own)."""
    P: list[str] = []
    diag: dict = {}
    tier = record.get("tier")
    spec = TIERS.get(tier)
    m = record.get("meta") or {}
    if spec is None:
        return [f"unknown tier {tier!r}"], diag
    view = m.get("view")
    if view not in VIEWS:
        return [f"unknown view {view!r}"], diag
    n = spec["n"]
    labels = list(LETTERS[:n])

    # ---- declared form -------------------------------------------------------
    if record.get("answer_type") != "choice" or record.get("choices") != labels:
        P.append(f"choices must be {labels}")
    question = record.get("question")
    if question != expected_question(view, n):
        P.append("question text does not match the declared form for this view/tier")
    if m.get("labels") != labels:
        P.append("meta.labels disagree with choices")
    answer = record.get("answer")
    if answer not in labels:
        return P + [f"answer {answer!r} not among {labels}"], diag
    if isinstance(question, str) and example_overlaps(question, answer):
        P.append("a format example in the question equals or shares a component with the answer")
    ans = labels.index(answer)
    if m.get("correct_index") != ans:
        P.append("meta.correct_index disagrees with the answer letter")

    # ---- meta structure -----------------------------------------------------
    try:
        H = np.asarray(m["heightmap"], dtype=int)
        if H.ndim != 2 or (H < 0).any():
            raise ValueError
        options = [_as_grid(g) for g in m["option_grids"]]
    except Exception:  # noqa: BLE001
        return P + ["heightmap/option_grids malformed"], diag
    N, M = H.shape
    if "bases" in spec:
        if (N, M) not in spec["bases"]["top" if view == "top" else "lateral"]:
            P.append(f"base {N}x{M} outside the {tier} tier for a {view} view")
    elif not (spec["rows"][0] <= N <= spec["rows"][1] and spec["cols"][0] <= M <= spec["cols"][1]):
        P.append(f"base {N}x{M} outside the {tier} tier")
    if H.max() > spec["hmax"] or H.max() < 1:
        P.append(f"heights outside 1..{spec['hmax']}")
    if m.get("tier_hmax") != spec["hmax"]:
        P.append("meta.tier_hmax disagrees with the tier")
    if len(options) != n:
        return P + [f"{len(options)} option grids, expected {n}"], diag
    # Lateral frame: as tall as the tallest bar of any option (so the truth need not reach the top
    # row), never taller than the tier allows.  Top views: the N x M base.
    if view == "top":
        F = int(H.max())
        want_shape = (N, M)
    else:
        F = options[0].shape[0]
        want_shape = (F, M if view == "front" else N)
    if any(g.shape != want_shape for g in options):
        P.append(f"option grids are not all {want_shape}")
        return P, diag
    if view != "top":
        if not all(np.all(np.diff(g, axis=0) >= 0) for g in options):
            P.append("a lateral option is not a bottom-filled silhouette")
        if not 1 <= F <= spec["hmax"] or max(int(g.sum(axis=0).max()) for g in options) != F:
            P.append("lateral frame is not the tallest bar of the options (within the tier's height)")
        if int(H.max()) > F:
            P.append("the figure is taller than the option frame")
    if m.get("frame_height") != F:
        P.append("meta.frame_height disagrees with the frame rule")
    if len({g.tobytes() for g in options}) != n:
        P.append("option grids are not distinct")
    # The grid may extend past the figure, so options are compared by shape: no two may share one.
    for a, b in itertools.combinations(range(n), 2):
        if is_translate(options[a], options[b]):
            P.append(f"options {labels[a]} and {labels[b]} are the same shape translated inside the frame")

    # ---- gold from first principles ---------------------------------------------
    size = spec["size"]
    lay = m.get("layout") or {}
    fig = lay.get("figure") or {}
    px = int(fig.get("size", 0))
    if px != int(size * FIGURE_FRACTION):
        P.append("figure size is not 58% of the image")
    margin = spec.get("margins", {}).get((N, M), 46)     # 46 px; easy 4x5 / 5x5: 38 / 30 (4x4 cube size)
    if fig.get("supersample") != SS or fig.get("margin") != margin:
        P.append("figure supersample/margin differ from the layout rule")
    cam = fit_camera(H, px, margin=margin, ss=SS)
    E = world_frame(*cam[2:])
    truth = project(H, view, F, E)
    if "true_projection" in m and not np.array_equal(np.asarray(m["true_projection"]), truth):
        P.append("meta.true_projection disagrees with the recomputed projection")
    match = [k for k, g in enumerate(options) if np.array_equal(g, truth)]
    if match != [ans]:
        P.append(f"options equal to the recomputed view: {[labels[k] for k in match]}, answer {answer}")
    if any(not g.any() for g in options):
        P.append("an option is empty")
    if match == [ans]:
        P += tier_problems(spec, view, H, options, truth, (record.get("source") or {}).get("index"))

    # ---- image binding --------------------------------------------------------
    img = np.asarray(image.convert("RGB"))
    if img.shape[:2] != (size, size):
        return P + [f"image is {img.shape[1]}x{img.shape[0]}, expected {size}"], diag
    fx, fy = (int(v) for v in fig.get("offset", (0, 0)))
    if (fx, fy) != ((size - px) // 2, 6):
        P.append("figure offset differs from the layout rule")
    color = tuple(int(c) for c in fig.get("color", PANEL_FILL))
    if color != PANEL_FILL:
        P.append("figure colour differs from the layout rule")
    ref = render(H, px, cam, color)
    crop = img[fy:fy + px, fx:fx + px]
    differs = np.any(crop != ref, axis=2)
    if (differs & np.any(ref != 255, axis=2)).any():
        P.append(f"{int((differs & np.any(ref != 255, axis=2)).sum())} figure px differ from the re-render of "
                 "meta.heightmap (or an overlay covers figure content)")
    ignore = differs  # overlay pixels: whatever lies under them is not visible
    arrow_meta = lay.get("arrow") or {}
    tl = lay.get("text_labels") or []
    panels = lay.get("panels") or {}
    boxes = panels.get("boxes") or []
    try:
        if (list(arrow_meta.get("color", [])) != list(ARROW_RGB) or arrow_meta.get("width") != ARROW_WIDTH
                or arrow_meta.get("head") != ARROW_HEAD):
            raise ValueError("arrow colour/width/head differ from the layout rule")
        base_layout = bool(panels.get("slots"))
        want = BASE_LETTER if base_layout else EXTRA_LETTER
        if (int(panels["label_font_size"]) != want["font_size"] or tuple(panels["label_color"]) != want["color"]
                or int(panels["pad"]) != want["pad"] or (base_layout and len(panels["slots"]) != n)):
            raise ValueError("option-zone fonts/padding differ from the layout rule")
        if len(boxes) != n or len(panels.get("label_xy") or []) != n or len(tl) != 1 or not tl[0]:
            raise ValueError("layout lists the wrong number of panels, letters or labels")
        overlay = overlay_mask(lay, size)
        # Every pixel of the image must be exactly the drawing of meta: the figure re-render, the
        # recorded arrow and word, the panels of meta.option_grids, the letters, slot outlines and
        # divider.  Nothing else (a decoy arrow, a faint mark, a recoloured label box) can pass.
        exp = expected_image(lay, labels, options, ref, size)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return P + [f"layout cannot be drawn: {exc}"], diag
    wrong = np.any(img != exp, axis=2)
    if wrong.any():
        ys, xs = np.nonzero(wrong)
        P.append(f"{int(wrong.sum())} px differ from the drawing of meta (figure, arrow, label, panels, letters, "
                 f"outlines; first at x={xs[0]}, y={ys[0]})")
    ov = overlay[fy:fy + px, fx:fx + px]
    if (ov & np.any(ref != 255, axis=2)).any():
        P.append("the arrow, label or divider overlaps the figure")
    hw, hh, ch = cam[2:]
    origin = np.array([fx + cam[0] / SS, fy + cam[1] / SS])        # world (0,0,0) in image px
    floor_mid = origin + screen_dir(((M + 1) / 2, (N + 1) / 2, 0), hw, hh, ch) / SS
    counts = surface_counts(H, px, cam) if surface_guard else np.full((1, 1), MIN_SURFACE)
    if counts.min() < MIN_SURFACE:
        P.append(f"a column top / floor cell shows only {int(counts.min())} robust px (< {MIN_SURFACE})")

    # arrow -> asked view (decoded from pixels, and the recorded segment must agree)
    arrow = decode_arrow(img)
    if arrow is None:
        P.append("no arrow found")
    else:
        angles = {v: _angle(arrow["dir"], screen_dir(VIEW_DIRECTION[v], hw, hh, ch)) for v in VIEWS}
        decoded = min(angles, key=angles.get)
        diag["arrow_angle_error"] = round(angles[view], 2)
        if decoded != view or angles[view] > 10:
            P.append(f"arrow points along the {decoded} direction (error {angles[decoded]:.1f} deg), "
                     f"question asks the {view} view")
        if (floor_mid - arrow["tail"]) @ arrow["dir"] <= 0:
            P.append("arrow does not point at the figure")
        (ax, ay), (bx, by) = arrow_meta["segment"]
        seg = np.array([bx - ax, by - ay], float)
        if _angle(seg, arrow["dir"]) > 10 or _angle(seg, screen_dir(VIEW_DIRECTION[view], hw, hh, ch)) > 10:
            P.append("meta.layout.arrow disagrees with the drawn arrow or the asked view")

    # FRONT / RIGHT label
    want_word = {"front": "FRONT", "side": "RIGHT", "top": "FRONT"}[view]
    if tl[0].get("text") != want_word or int(tl[0].get("font_size", 0)) != 10:
        P.append(f"expected one {want_word} label")
    else:
        exp_l, ink = _text_ink((size, size), tl[0]["xy"], want_word, 10, ARROW_RGB)
        if not ink.any() or np.any(img[ink] != exp_l[ink]):
            P.append(f"{want_word} label pixels do not match")
        ys, xs = np.nonzero(ink)
        if len(xs):
            lc = np.array([xs.mean(), ys.mean()])
            if view == "top":
                rel = lc - floor_mid
                pj = rel @ screen_dir((0, 1, 0), hw, hh, ch) / np.linalg.norm(screen_dir((0, 1, 0), hw, hh, ch))
                pi = rel @ screen_dir((1, 0, 0), hw, hh, ch) / np.linalg.norm(screen_dir((1, 0, 0), hw, hh, ch))
                # closest to +j of the four floor sides +-i, +-j (below-left of the centre)
                if not pj > abs(pi):
                    P.append("FRONT label is not on the +j (front) side of the figure")
            elif arrow is not None and np.linalg.norm(lc - arrow["tail"]) > 45:
                P.append(f"{want_word} label is not at the arrow's tail")

    # option panels and letters, decoded from pixels
    cell, pad = int(panels.get("cell", 0)), int(panels.get("pad", 0))
    for k, box in enumerate(boxes):
        got = decode_panel(img, box, cell, pad)
        if isinstance(got, str):
            P.append(got)
        elif not np.array_equal(got, options[k]):
            P.append(f"panel {labels[k]} decoded {got.tolist()} != meta option {options[k].tolist()}")
    for k, xy in enumerate(panels["label_xy"]):
        exp_t, ink = _text_ink((size, size), xy, labels[k], int(panels["label_font_size"]), panels["label_color"])
        mismatch = ink & np.any(img != exp_t, axis=2)
        if not ink.any() or mismatch.sum() > 0:
            P.append(f"letter {labels[k]} not drawn as expected ({int(mismatch.sum())} px differ)")
        ys, xs = np.nonzero(ink)
        if len(xs):
            # a letter names the panel in the nearest row below it with the nearest centre
            below = [b for b in boxes if b[1] > ys.max()]
            row = [b for b in below if b[1] == min((c[1] for c in below), default=None)]
            near = min(row, key=lambda b: abs((b[0] + b[2]) / 2 - xs.mean()), default=None)
            if near is None or list(near) != list(boxes[k]):
                P.append(f"letter {labels[k]} is not attached to panel {labels[k]}")

    # ---- uniqueness -------------------------------------------------------------
    if uniqueness and not P:
        up, stats = analyse_uniqueness(H, view, truth, options, ans, max(spec["hmax"], F), cam, px, color,
                                       ignore, E)
        P += up
        diag.update(stats)
        if exhaustive and not P:
            found, st = exhaustive_refutation(H, view, options, ans, cam, px, color, ignore, E, ref)
            diag.update(exhaustive_nodes=st["nodes"], exhaustive_targets=st.get("targets"))
            if st["aborted"]:
                P.append(f"exhaustive uniqueness search did not finish ({st.get('reason', 'node budget')}); "
                         "the item is refused rather than left unproven")
            elif found is not None:
                k, Wx = found
                P.append(f"option {LETTERS[k]} not robustly refuted: the scene {Wx.tolist()} on the drawn grid "
                         "(base plus front padding strip) has that option's shape and renders without a robust "
                         "difference")
    return P, diag
