"""Procedural braided-cord glyphs. No image-generation model is used."""

import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
def skeletonize(binary):
    """Deterministic Zhang-Suen thinning; no optional imaging package needed."""
    image = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            center = image[1:-1, 1:-1]
            p = (image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
                 image[2:, 2:], image[2:, 1:-1], image[2:, :-2],
                 image[1:-1, :-2], image[:-2, :-2])
            neighbors = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] != 0)).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                condition = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                condition = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = ((center != 0) & (neighbors >= 2) & (neighbors <= 6)
                      & (transitions == 1) & condition)
            if remove.any():
                center[remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def _stroke_coordinates(binary):
    """Distance along the stroke, signed distance across it, and local radius."""
    skel = skeletonize(binary)
    yy, xx = np.nonzero(skel)
    if not len(yy):
        return None
    index = np.full(binary.shape, -1, dtype=np.int32)
    index[yy, xx] = np.arange(len(yy))
    rows, cols, costs = [], [], []
    height, width = binary.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ny, nx = yy + dy, xx + dx
        ok = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        source = np.flatnonzero(ok)
        target = index[ny[ok], nx[ok]]
        keep = target >= 0
        source, target = source[keep], target[keep]
        rows.extend((source, target))
        cols.extend((target, source))
        costs.extend((np.full(len(source), math.hypot(dy, dx)),) * 2)
    graph = coo_matrix(
        (np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(yy), len(yy)),
    ).tocsr()
    n_components, component = connected_components(graph)
    roots = [np.flatnonzero(component == c)[0] for c in range(n_components)]
    along_nodes = np.min(dijkstra(graph, directed=False, indices=roots), axis=0)
    along_nodes[~np.isfinite(along_nodes)] = 0
    along = np.zeros(binary.shape, dtype=float)
    along[yy, xx] = along_nodes
    # A locally averaged arc coordinate gives a consistently oriented tangent;
    # this avoids a global diagonal hatch and follows curved letter strokes.
    stroke_distance = ndi.distance_transform_edt(binary)
    sigma = max(.6, float(np.median(stroke_distance[skel])) / 6.0)
    support = ndi.gaussian_filter(skel.astype(float), sigma)
    smooth = ndi.gaussian_filter(along, sigma) / np.maximum(support, 1e-7)
    tangent_y, tangent_x = np.gradient(smooth)
    _, nearest = ndi.distance_transform_edt(~skel, return_indices=True)
    sy, sx = nearest
    ty, tx = tangent_y[sy, sx], tangent_x[sy, sx]
    norm = np.hypot(tx, ty)
    ty, tx = ty / np.maximum(norm, 1e-7), tx / np.maximum(norm, 1e-7)
    tx[norm < 1e-7] = 1.0
    y, x = np.indices(binary.shape)
    delta_y, delta_x = y - sy, x - sx
    s = along[sy, sx] + delta_y * ty + delta_x * tx
    across = -delta_x * ty + delta_y * tx
    radius = stroke_distance[sy, sx]
    radius = np.maximum(radius, .75)
    return s, across, radius, tx, ty


def render_rope(mask: Image.Image, rgb: tuple, seed: int) -> Image.Image:
    """Render three interwoven, fiber-textured cords along a grayscale glyph.

    Returns a transparent RGBA image exactly the same size as ``mask``.
    The clean mask remains the support, preserving holes and stroke junctions.
    """
    alpha = np.asarray(mask.convert("L"), dtype=float) / 255.0
    binary = alpha > 0.5
    if not binary.any():
        return Image.new("RGBA", mask.size)
    coordinates = _stroke_coordinates(binary)
    if coordinates is None:
        return Image.new("RGBA", mask.size)
    s, across, radius, tx, ty = coordinates
    rng = np.random.default_rng(seed)
    base = np.asarray(rgb, dtype=float) / 255.0
    median_radius = float(np.median(radius[binary]))
    scale = max(.15, median_radius / 20.0)
    pitch = max(9.0, median_radius * 4.2) * rng.uniform(.92, 1.08)
    phase = 2.0 * np.pi * s / pitch + rng.uniform(0, 2 * np.pi)
    # A darker cord interior keeps the silhouette readable between front strands.
    core = np.sqrt(np.clip(1.0 - (across / radius) ** 2, 0, 1))
    shade = 0.44 + 0.23 * core
    canvas = base[None, None, :] * shade[..., None]
    best_depth = np.full(alpha.shape, -1e9)
    smooth_noise = ndi.gaussian_filter(rng.normal(size=alpha.shape), max(.35, .6 * scale))
    for strand in range(3):
        p = phase + strand * 2.0 * np.pi / 3.0
        # Elliptical helical strands: their front/back order changes along s.
        center = 0.58 * radius * np.cos(p)
        strand_radius = 0.47 * radius
        q = (across - center) / strand_radius
        inside = np.abs(q) < 1.0
        cap = np.sqrt(np.clip(1.0 - q * q, 0, 1))
        depth = 0.58 * radius * np.sin(p) + strand_radius * cap
        front = inside & (depth > best_depth)
        best_depth = np.where(front, depth, best_depth)
        # Fine twisted yarns run WITH the changing strand, not across the page.
        threads = np.cos(2 * np.pi * (q * 5.8 + s / (115.0 * scale) + 0.16 * np.sin(p)))
        fine_threads = np.cos(2 * np.pi * (q * 14.0 + s / (43.0 * scale)))
        cross_light = q * (0.65 * ty - 0.45 * tx)
        lighting = (
            0.52 + 0.49 * cap + 0.12 * cross_light
            + 0.065 * threads + 0.027 * fine_threads
            + 0.040 * smooth_noise
        )
        # Back-facing cord sections are slightly darker at each braid crossing.
        lighting *= 0.86 + 0.14 * (np.sin(p) + 1.0) / 2.0
        color = base[None, None, :] * lighting[..., None]
        highlight = 0.11 * np.maximum(cap - 0.45, 0)[..., None]
        color += (1.0 - base)[None, None, :] * highlight
        canvas = np.where(front[..., None], color, canvas)
    # Tiny fiber-level variation, kept inside the existing glyph support.
    canvas = np.clip(canvas, 0, 1)
    result = np.dstack((canvas, alpha))
    return Image.fromarray(np.round(result * 255).astype(np.uint8), "RGBA")
