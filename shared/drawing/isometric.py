"""Shared isometric cube-stack renderer for the 3D generators.

Coordinate system
-----------------
A heightmap H is an N(rows, depth index j) x M(cols, width index i) integer
array. H[j][i] = number of unit cubes stacked at base cell (i, j). Cubes are
gravity-filled (a stack of height h occupies levels 0..h-1, no floats).

Isometric projection (classic 2:1 "video-game" iso):
  Screen-x = ORIGIN_X + (i - j) * HW
  Screen-y = ORIGIN_Y + (i + j) * HH - k * CH
where i = column (0..M-1), j = row/depth (0..N-1), k = cube level.
HW = half-tile width, HH = half-tile height, CH = cube vertical height (px).

With this mapping:
  * increasing i  -> moves right & down on screen (front-right direction)
  * increasing j  -> moves left  & down on screen (front-left direction)
  * the cell with the LARGEST (i + j) is closest to the viewer (front).
So painter's order (back -> front) sorts cells by ascending (i + j); for equal
(i+j) order does not matter (they don't overlap). Within a column we draw from
bottom level up. This yields correct occlusion for any gravity-filled stack.

Faces & tones (light gray):
  top   = lightest, left = medium-dark, right = medium. Black edges.

Camera: placement() centres and scales H on the supersampled canvas;
render_fixed() draws any heightmap with a given camera, so an edited structure
keeps the pose of the original; render_stack() is placement + render_fixed.
"""
from __future__ import annotations

from PIL import Image, ImageDraw


# ---- default isometric metrics (px) ----
HW = 34   # half tile width
HH = 17   # half tile height  (2:1 iso)
CH = 40   # cube vertical height
SHADES = (1.0, 0.74, 0.88)  # top, left (+j, front-left) and right (+i, front-right) faces


def iso_xy(i, j, k, origin, hw=HW, hh=HH, ch=CH):
    ox, oy = origin
    x = ox + (i - j) * hw
    y = oy + (i + j) * hh - k * ch
    return (x, y)


def _shade(base, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in base)


def cube_polys(i, j, k, origin, hw=HW, hh=HH, ch=CH):
    """Return (top, left, right) face polygons for the cube at (i,j) level k.

    top   : the upper rhombus
    left  : the face toward +j (front-left), i.e. the face you see on the left
    right : the face toward +i (front-right)
    """
    # 8 corners of the cube
    # base level k (bottom), top level k+1
    def P(di, dj, dk):
        return iso_xy(i + di, j + dj, k + dk, origin, hw, hh, ch)

    # top face corners (at level k+1)
    t00 = P(0, 0, 1)   # back corner
    t10 = P(1, 0, 1)   # right corner
    t11 = P(1, 1, 1)   # front corner
    t01 = P(0, 1, 1)   # left corner
    top = [t00, t10, t11, t01]

    # right face (i = i+1 plane), toward viewer's right
    rb_top = t10            # top-back of right face
    rf_top = t11            # top-front
    rf_bot = P(1, 1, 0)
    rb_bot = P(1, 0, 0)
    right = [rb_top, rf_top, rf_bot, rb_bot]

    # left face (j = j+1 plane), toward viewer's left
    lb_top = t01
    lf_top = t11
    lf_bot = P(1, 1, 0)
    lb_bot = P(0, 1, 0)
    left = [lb_top, lf_top, lf_bot, lb_bot]

    return top, left, right


def draw_cube(draw: ImageDraw.ImageDraw, i, j, k, origin, base_rgb=(205, 208, 214),
              hw=HW, hh=HH, ch=CH, edge=(20, 20, 20), edge_w=2, shades=SHADES):
    top, left, right = cube_polys(i, j, k, origin, hw, hh, ch)
    c_top = _shade(base_rgb, shades[0])
    c_left = _shade(base_rgb, shades[1])
    c_right = _shade(base_rgb, shades[2])
    draw.polygon(top, fill=c_top, outline=edge)
    draw.polygon(left, fill=c_left, outline=edge)
    draw.polygon(right, fill=c_right, outline=edge)
    # redraw outlines so face fills don't cover the shared seams
    for poly in (top, left, right):
        draw.line(poly + [poly[0]], fill=edge, width=edge_w)


def draw_grid_floor(draw: ImageDraw.ImageDraw, N, M, origin,
                    hw=HW, hh=HH, ch=CH, color=(200, 205, 212), w=1, pad=1):
    """Thin isometric grid floor (level 0), N rows (j) by M cols (i)."""
    n = N + pad
    m = M + pad
    # lines of constant i (vary j)
    for i in range(m + 1):
        a = iso_xy(i, 0, 0, origin, hw, hh, ch)
        b = iso_xy(i, n, 0, origin, hw, hh, ch)
        draw.line([a, b], fill=color, width=w)
    for j in range(n + 1):
        a = iso_xy(0, j, 0, origin, hw, hh, ch)
        b = iso_xy(m, j, 0, origin, hw, hh, ch)
        draw.line([a, b], fill=color, width=w)


def heightmap_bbox(H, origin, hw=HW, hh=HH, ch=CH, pad_floor=1):
    """Compute screen bbox of everything drawn (floor + cubes)."""
    N = len(H)
    M = len(H[0])
    xs, ys = [], []
    # floor corners
    for (i, j) in [(0, 0), (M + pad_floor, 0), (0, N + pad_floor), (M + pad_floor, N + pad_floor)]:
        x, y = iso_xy(i, j, 0, origin, hw, hh, ch)
        xs.append(x); ys.append(y)
    for j in range(N):
        for i in range(M):
            h = H[j][i]
            if h <= 0:
                continue
            for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                x, y = iso_xy(i + di, j + dj, h, origin, hw, hh, ch)
                xs.append(x); ys.append(y)
                x2, y2 = iso_xy(i + di, j + dj, 0, origin, hw, hh, ch)
                xs.append(x2); ys.append(y2)
    return min(xs), min(ys), max(xs), max(ys)


def placement(H, size, margin=46, supersample=2):
    """Camera of render_stack: H centred and scaled to fit the supersampled canvas.

    Returns info = dict(N, M, hw, hh, ch, origin, ss).  The bbox always includes
    a one-cell floor pad (heightmap_bbox pad_floor=1), whatever floor is drawn,
    so the camera depends only on H, size, margin and supersample.
    """
    ss = supersample
    hw, hh, ch = HW * ss, HH * ss, CH * ss
    x0, y0, x1, y1 = heightmap_bbox(H, (0, 0), hw, hh, ch)
    w = x1 - x0
    h = y1 - y0
    canvas = size * ss
    avail = canvas - 2 * margin * ss
    # scale so the structure fits (keep iso metrics integer-ish; scale origin only)
    scale = min(1.0, avail / max(w, h, 1))
    hw = max(2, int(hw * scale)); hh = max(1, int(hh * scale)); ch = max(2, int(ch * scale))
    x0, y0, x1, y1 = heightmap_bbox(H, (0, 0), hw, hh, ch)
    w = x1 - x0; h = y1 - y0
    ox = (canvas - w) / 2 - x0
    oy = (canvas - h) / 2 - y0
    return dict(N=len(H), M=len(H[0]), hw=hw, hh=hh, ch=ch, origin=(ox, oy), ss=ss)


def draw_fixed(H, size, info, base_rgb=(205, 208, 214), bg=(255, 255, 255),
               floor_color=(200, 205, 212), draw_floor=True, floor_pad=1, shades=SHADES):
    """Draw H with the camera `info` (from placement) on the supersampled canvas."""
    N = len(H)
    M = len(H[0])
    ss = info["ss"]
    hw, hh, ch, origin = info["hw"], info["hh"], info["ch"], info["origin"]
    img = Image.new("RGB", (size * ss, size * ss), bg)
    draw = ImageDraw.Draw(img)
    if draw_floor:
        draw_grid_floor(draw, N, M, origin, hw, hh, ch, color=floor_color,
                        w=max(1, ss), pad=floor_pad)
    # painter's algorithm: ascending (i + j), then ascending k
    cells = [(i, j) for j in range(N) for i in range(M)]
    cells.sort(key=lambda c: (c[0] + c[1]))
    for (i, j) in cells:
        for k in range(H[j][i]):
            draw_cube(draw, i, j, k, origin, base_rgb=base_rgb,
                      hw=hw, hh=hh, ch=ch, edge=(20, 20, 20), edge_w=max(1, ss),
                      shades=shades)
    return img


def render_fixed(H, size, info, **kwargs):
    """draw_fixed downsampled to size x size, so edits of H cannot recenter the image."""
    img = draw_fixed(H, size, info, **kwargs)
    if info["ss"] != 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def render_stack(H, size=448, base_rgb=(205, 208, 214), bg=(255, 255, 255),
                 floor_color=(200, 205, 212), margin=46, draw_floor=True,
                 supersample=2, floor_pad=1, shades=SHADES):
    """Render the gravity-filled heightmap H to a centered RGB image.

    H[j][i] = stack height at base cell (i col, j depth-row).
    Returns (PIL.Image, info) where info has the placement metrics.
    floor_pad=1 (default) draws one extra floor row/column in front of the
    grid; floor_pad=0 draws exactly the N x M footprint grid.
    """
    info = placement(H, size, margin, supersample)
    img = render_fixed(H, size, info, base_rgb=base_rgb, bg=bg, floor_color=floor_color,
                       draw_floor=draw_floor, floor_pad=floor_pad, shades=shades)
    return img, info
