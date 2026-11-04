"""Triangle-grid ("polytan") shape helpers of tangram_reconstruction.

Shapes live on the quarter-square triangle grid (each unit cell split by both
diagonals into 4 triangles), so targets and pieces have diagonal (45-degree)
edges like tangram pieces instead of being polyominoes.

Grid representation
-------------------
A triangle cell is (x, y, t) where (x, y) is the unit square and
t in {0=N, 1=E, 2=S, 3=W} names the quarter facing that edge (y grows down).
90-degree clockwise rotation maps (x, y, t) -> (-y - 1, x, (t + 1) % 4).
45-degree rotations can never map a lattice polytan onto the lattice
(vertices leave (1/2)Z^2), so "rotations in 90-degree steps" is exhaustive.

sample_polytan grows a random connected shape; split_target partitions a
target into connected parts (random multi-source BFS on the triangle adjacency
graph); draw_tris rasterises a shape.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Polytan helpers (cells = frozenset of (x, y, t); t: 0=N 1=E 2=S 3=W, y down)
# --------------------------------------------------------------------------- #
def normalize(cells):
    cells = list(cells)
    x0 = min(x for x, _, _ in cells)
    y0 = min(y for _, y, _ in cells)
    return frozenset((x - x0, y - y0, t) for x, y, t in cells)


def rot90(cells):
    return normalize((-y - 1, x, (t + 1) % 4) for x, y, t in cells)


def neighbors(cell):
    """Edge-adjacent triangle cells: two within the square, one across."""
    x, y, t = cell
    across = ((x, y - 1, 2), (x + 1, y, 3), (x, y + 1, 0), (x - 1, y, 1))[t]
    return ((x, y, (t + 1) % 4), (x, y, (t + 3) % 4), across)


def connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        for nb in neighbors(c):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def sample_polytan(rng, n_tris):
    """Random connected polytan grown triangle by triangle, biased toward
    completing unit squares so silhouettes stay chunky/tangram-like."""
    cells = {(0, 0, rng.randrange(4))}
    while len(cells) < n_tris:
        frontier = set()
        for c in cells:
            for nb in neighbors(c):
                if nb not in cells:
                    frontier.add(nb)
        cand = sorted(frontier)
        filled = {}
        for x, y, t in cells:
            filled[(x, y)] = filled.get((x, y), 0) + 1
        weights = [1 + 2 * filled.get((x, y), 0) for x, y, t in cand]
        cells.add(rng.choices(cand, weights=weights)[0])
    return normalize(cells)


def split_target(rng, target, n_parts, min_size=4):
    """Random multi-source BFS partition into connected pieces."""
    cells = sorted(target)
    for _ in range(60):
        seeds = rng.sample(cells, n_parts)
        owner = {s: k for k, s in enumerate(seeds)}
        frontier = list(seeds)
        while frontier:
            k = rng.randrange(len(frontier))
            c = frontier.pop(k)
            for nb in neighbors(c):
                if nb in target and nb not in owner:
                    owner[nb] = owner[c]
                    frontier.append(nb)
        parts = [frozenset(c for c, o in owner.items() if o == k)
                 for k in range(n_parts)]
        if all(len(p) >= min_size and connected(p) for p in parts):
            return parts
    return None


# --------------------------------------------------------------------------- #
# Problem building
# --------------------------------------------------------------------------- #
RUNGS = {
    1:dict(n_parts=4,tris=(28,36),rotations=True,decoy="swap"),
    2:dict(n_parts=5,tris=(36,44),rotations=True,decoy="move1"),
    3:dict(n_parts=6,tris=(44,52),rotations=True,decoy="move1"),
    4:dict(n_parts=7,tris=(52,60),rotations=True,decoy="move1"),
    5:dict(n_parts=8,tris=(60,68),rotations=True,decoy="move1"),
}


EDGE_COLOR = (40, 40, 44)


def tri_vertices(cell):
    """Triangle corners in doubled cell units (so everything is an int)."""
    x, y, t = cell
    c = (2 * x + 1, 2 * y + 1)
    corners = ((2 * x, 2 * y), (2 * x + 2, 2 * y),
               (2 * x + 2, 2 * y + 2), (2 * x, 2 * y + 2))
    return (corners[t], corners[(t + 1) % 4], c)


def draw_tris(d, cells, cell_px, ox, oy, fill):
    """Fill the union of triangles and stroke only its boundary edges."""
    def px(v):
        return (ox + v[0] * cell_px / 2.0, oy + v[1] * cell_px / 2.0)

    edge_count = {}
    for c in sorted(cells):
        vs = tri_vertices(c)
        d.polygon([px(v) for v in vs], fill=fill)
        for i in range(3):
            e = frozenset((vs[i], vs[(i + 1) % 3]))
            edge_count[e] = edge_count.get(e, 0) + 1
    for e, n in edge_count.items():
        if n == 1:
            a, b = sorted(e)
            d.line([px(a), px(b)], fill=EDGE_COLOR, width=2)


def bbox_wh(cells):
    return (max(x for x, _, _ in cells) + 1, max(y for _, y, _ in cells) + 1)
