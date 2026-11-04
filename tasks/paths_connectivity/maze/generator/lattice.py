"""Maze lattices: carving, graph search, styles and drawing.

Inspired by "Thinking with Visual Primitives" (Sec. 2.4.3). Three maze
*topologies* (lattices) are supported: rectangular grids, hexagonal honeycomb
lattices, and circular concentric-ring mazes.

The maze carving (DFS / Prim / Kruskal / Wilson) is written once against an
abstract neighbour function, so every algorithm works on every lattice. Ground
truth is always recomputed with BFS on the final wall set, so labels cannot
drift from what is actually drawn.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

from PIL import Image, ImageChops, ImageDraw

from shared.drawing.fonts import load_bundled_font

from shared.drawing.geometry import Cell, Wall
NeighborFn = Callable[[Cell], Iterable[Cell]]


def wall_key(a: Cell, b: Cell) -> Wall:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Generic graph utilities (keyed on a neighbour function + closed-wall set)
# --------------------------------------------------------------------------- #
def all_walls(cells: list[Cell], neighbor_fn: NeighborFn) -> set[Wall]:
    walls: set[Wall] = set()
    for c in cells:
        for nb in neighbor_fn(c):
            walls.add(wall_key(c, nb))
    return walls


def reachable(start: Cell, neighbor_fn: NeighborFn, walls: set[Wall]) -> set[Cell]:
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in neighbor_fn(cur):
            if nb in seen or wall_key(cur, nb) in walls:
                continue
            seen.add(nb)
            queue.append(nb)
    return seen


def components(cells: list[Cell], neighbor_fn: NeighborFn, walls: set[Wall]) -> list[set[Cell]]:
    remaining = set(cells)
    out: list[set[Cell]] = []
    while remaining:
        comp = reachable(next(iter(remaining)), neighbor_fn, walls)
        out.append(comp)
        remaining -= comp
    return out


def shortest_path(start: Cell, goal: Cell, neighbor_fn: NeighborFn, walls: set[Wall]) -> list[Cell] | None:
    prev: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nb in neighbor_fn(cur):
            if nb in prev or wall_key(cur, nb) in walls:
                continue
            prev[nb] = cur
            queue.append(nb)
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def bfs_dist(start: Cell, neighbor_fn: NeighborFn,
             walls: set[Wall]) -> dict:
    """BFS distances from `start` over the reachable set.

    Mirrors exactly how ``reachable`` / ``shortest_path`` decide a wall
    blocks two adjacent cells (``wall_key`` membership in ``walls``).
    """
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        d = dist[cur]
        for nb in neighbor_fn(cur):
            if nb in dist or wall_key(cur, nb) in walls:
                continue
            dist[nb] = d + 1
            queue.append(nb)
    return dist


def open_degree(cell: Cell, neighbor_fn: NeighborFn,
                walls: set[Wall]) -> int:
    """Number of immediately open passages out of a cell."""
    return sum(wall_key(cell, nb) not in walls for nb in neighbor_fn(cell))


# --------------------------------------------------------------------------- #
# Maze carving algorithms (perfect mazes = spanning trees; return closed walls)
# --------------------------------------------------------------------------- #
def carve_dfs(cells: list[Cell], neighbor_fn: NeighborFn, rng: random.Random) -> set[Wall]:
    walls = all_walls(cells, neighbor_fn)
    start = rng.choice(cells)
    stack, seen = [start], {start}
    while stack:
        cur = stack[-1]
        choices = [nb for nb in neighbor_fn(cur) if nb not in seen]
        if not choices:
            stack.pop()
            continue
        nxt = rng.choice(choices)
        walls.discard(wall_key(cur, nxt))
        seen.add(nxt)
        stack.append(nxt)
    return walls


def carve_prim(cells: list[Cell], neighbor_fn: NeighborFn, rng: random.Random) -> set[Wall]:
    walls = all_walls(cells, neighbor_fn)
    start = rng.choice(cells)
    seen = {start}
    frontier = [(start, nb) for nb in neighbor_fn(start)]
    while frontier:
        a, b = frontier.pop(rng.randrange(len(frontier)))
        if b in seen:
            continue
        walls.discard(wall_key(a, b))
        seen.add(b)
        frontier.extend((b, nb) for nb in neighbor_fn(b) if nb not in seen)
    return walls


def carve_kruskal(cells: list[Cell], neighbor_fn: NeighborFn, rng: random.Random) -> set[Wall]:
    parent = {c: c for c in cells}

    def find(c: Cell) -> Cell:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    walls = all_walls(cells, neighbor_fn)
    candidates = list(walls)
    rng.shuffle(candidates)
    for a, b in candidates:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            walls.discard(wall_key(a, b))
    return walls


def carve_wilson(cells: list[Cell], neighbor_fn: NeighborFn, rng: random.Random) -> set[Wall]:
    walls = all_walls(cells, neighbor_fn)
    order = list(cells)
    rng.shuffle(order)
    in_tree = {order[0]}
    for target in order[1:]:
        if target in in_tree:
            continue
        path = [target]
        index = {target: 0}
        cur = target
        while cur not in in_tree:
            nxt = rng.choice(list(neighbor_fn(cur)))
            if nxt in index:
                drop = index[nxt] + 1
                for c in path[drop:]:
                    del index[c]
                del path[drop:]
            else:
                index[nxt] = len(path)
                path.append(nxt)
            cur = nxt
        for a, b in zip(path, path[1:]):
            walls.discard(wall_key(a, b))
            in_tree.add(a)
            in_tree.add(b)
        in_tree.add(path[-1])
    return walls


CARVERS = {"dfs": carve_dfs, "prim": carve_prim, "kruskal": carve_kruskal, "wilson": carve_wilson}


# --------------------------------------------------------------------------- #
# Styling: colour palette + line thickness (the same maze can be drawn any way)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Style:
    bg: tuple
    wall: tuple
    thickness: float = 1.0  # multiplier on each lattice's base line width


# (background, wall) per palette. Each keeps a high wall/background
# contrast so the maze stays readable after the SigLIP 448 downsample.
PALETTES = {
    "classic":   ((247, 249, 252), (24, 28, 38)),
    "ink":       ((244, 241, 232), (40, 34, 30)),
    "blueprint": ((26, 50, 92), (226, 233, 246)),
    "forest":    ((234, 245, 238), (26, 70, 46)),
    "slate":     ((236, 238, 241), (52, 60, 72)),
    "crimson":   ((250, 246, 246), (122, 28, 36)),
    "midnight":  ((18, 20, 28), (214, 219, 230)),
    "sand":      ((238, 228, 205), (92, 64, 38)),
    "rose":      ((250, 238, 243), (150, 40, 92)),
    "grape":     ((241, 236, 249), (74, 42, 112)),
}
HELDOUT_PALETTES = {
    "eval_lavender": ((245, 241, 252), (62, 48, 112)),
    "eval_mint":     ((235, 250, 244), (28, 91, 82)),
    "eval_charcoal": ((31, 34, 38), (226, 226, 214)),
    "eval_copper":   ((249, 241, 230), (130, 70, 34)),
    "eval_ice":      ((239, 249, 252), (45, 91, 125)),
}
ALL_PALETTES = {**PALETTES, **HELDOUT_PALETTES}
# Kept deliberately within a legible band — nothing hairline, nothing that
# closes the passages.
THICKNESS = {"thin": 0.7, "medium": 1.0, "thick": 1.5}


def make_style(palette: str = "classic", thickness: str = "medium") -> Style:
    bg, wall = ALL_PALETTES[palette]
    return Style(bg=bg, wall=wall, thickness=THICKNESS[thickness])


DEFAULT_STYLE = make_style()


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #
class Lattice:
    """A maze topology: cells, adjacency, and pixel geometry + rendering."""

    name: str
    cells: list[Cell]
    cellset: set[Cell]
    line_width: int       # base wall thickness for this lattice/size
    unit: float           # characteristic cell dimension, for clamping

    def styled_width(self, thickness: float) -> int:
        """Apply a thickness multiplier, clamped so walls stay legible and the
        passages never close (>=2px, <=30% of a cell)."""
        return max(2, min(round(self.line_width * thickness), round(self.unit * 0.30)))

    def neighbors(self, cell: Cell) -> list[Cell]:
        raise NotImplementedError

    def center(self, cell: Cell) -> tuple[float, float]:
        raise NotImplementedError

    def is_border(self, cell: Cell) -> bool:
        raise NotImplementedError

    def image_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def draw_walls(self, draw: ImageDraw.ImageDraw, walls: set[Wall],
                   color: tuple, line_width: int) -> None:
        raise NotImplementedError


class StraightLattice(Lattice):
    """Lattices whose walls are straight edges (rect, hex).

    Each direction has a constant pixel delta to the neighbour centre. A wall is
    drawn as the segment at the midpoint, perpendicular to that delta, of length
    `edge_len` — which reproduces exact square and hexagon edges. Boundary edges
    (no neighbour) are drawn too, giving a clean outline.
    """

    dir_pix: list[tuple[float, float]]
    edge_len: float
    _centers: dict[Cell, tuple[float, float]]

    def neighbor_in_dir(self, cell: Cell, i: int) -> Cell | None:
        raise NotImplementedError

    def neighbors(self, cell: Cell) -> list[Cell]:
        out = []
        for i in range(len(self.dir_pix)):
            nb = self.neighbor_in_dir(cell, i)
            if nb is not None:
                out.append(nb)
        return out

    def center(self, cell: Cell) -> tuple[float, float]:
        return self._centers[cell]

    def is_border(self, cell: Cell) -> bool:
        return any(self.neighbor_in_dir(cell, i) is None for i in range(len(self.dir_pix)))

    def draw_walls(self, draw: ImageDraw.ImageDraw, walls: set[Wall],
                   color: tuple, line_width: int) -> None:
        segments: set[tuple] = set()
        for cell in self.cells:
            cx, cy = self.center(cell)
            for i, (dx, dy) in enumerate(self.dir_pix):
                nb = self.neighbor_in_dir(cell, i)
                if nb is not None and wall_key(cell, nb) not in walls:
                    continue  # open passage
                mx, my = cx + dx / 2, cy + dy / 2
                length = math.hypot(dx, dy)
                px, py = -dy / length, dx / length
                h = self.edge_len / 2
                a = (round(mx + px * h, 1), round(my + py * h, 1))
                b = (round(mx - px * h, 1), round(my - py * h, 1))
                segments.add(tuple(sorted((a, b))))
        r = line_width / 2
        endpoints: set[tuple] = set()
        for a, b in segments:
            draw.line((a, b), fill=color, width=line_width)
            endpoints.add(a)
            endpoints.add(b)
        for (ex, ey) in endpoints:  # fill vertex joints so corners are crisp
            draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=color)


class RectLattice(StraightLattice):
    name = "rect"
    # order: E, W, S, N
    _offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, width: int, height: int, cell: int):
        self.width, self.height, self.cell = width, height, cell
        self.line_width = max(3, cell // 12)
        self.unit = cell
        self.margin = int(cell * 0.8)
        self.edge_len = cell
        self.dir_pix = [(dx * cell, dy * cell) for dx, dy in self._offsets]
        self.cells = [(x, y) for y in range(height) for x in range(width)]
        self.cellset = set(self.cells)
        m = self.margin
        self._centers = {
            (x, y): (m + x * cell + cell / 2, m + y * cell + cell / 2)
            for x, y in self.cells
        }

    def neighbor_in_dir(self, cell: Cell, i: int) -> Cell | None:
        x, y = cell
        dx, dy = self._offsets[i]
        nb = (x + dx, y + dy)
        return nb if nb in self.cellset else None

    def image_size(self) -> tuple[int, int]:
        return (self.margin * 2 + self.width * self.cell, self.margin * 2 + self.height * self.cell)


class HexLattice(StraightLattice):
    """Pointy-top honeycomb in odd-r offset coordinates (rectangular block)."""

    name = "hex"
    # direction order: E, W, NE, NW, SE, SW
    _S3 = math.sqrt(3)
    _cell_even = [(1, 0), (-1, 0), (0, -1), (-1, -1), (0, 1), (-1, 1)]
    _cell_odd = [(1, 0), (-1, 0), (1, -1), (0, -1), (1, 1), (0, 1)]

    def __init__(self, width: int, height: int, size: int):
        s = size
        s3 = self._S3
        self.width, self.height, self.size = width, height, size
        self.line_width = max(3, int(size / 3.5))
        self.unit = size
        self.edge_len = size  # hex side == centre-to-corner
        self.dir_pix = [
            (s3 * s, 0), (-s3 * s, 0),
            (s3 / 2 * s, -1.5 * s), (-s3 / 2 * s, -1.5 * s),
            (s3 / 2 * s, 1.5 * s), (-s3 / 2 * s, 1.5 * s),
        ]
        self.cells = [(c, r) for r in range(height) for c in range(width)]
        self.cellset = set(self.cells)
        raw = {
            (c, r): (s3 * s * c + (s3 / 2 * s if r & 1 else 0), 1.5 * s * r)
            for c, r in self.cells
        }
        xs = [p[0] for p in raw.values()]
        ys = [p[1] for p in raw.values()]
        margin = int(size * 0.9)
        off_x = margin + (s3 / 2 * s) - min(xs)
        off_y = margin + s - min(ys)
        self._centers = {k: (v[0] + off_x, v[1] + off_y) for k, v in raw.items()}
        self._img = (
            int(max(xs) - min(xs) + s3 * s + 2 * margin),
            int(max(ys) - min(ys) + 2 * s + 2 * margin),
        )

    def neighbor_in_dir(self, cell: Cell, i: int) -> Cell | None:
        c, r = cell
        dc, dr = (self._cell_odd if r & 1 else self._cell_even)[i]
        nb = (c + dc, r + dr)
        return nb if nb in self.cellset else None

    def image_size(self) -> tuple[int, int]:
        return self._img


class PolarLattice(Lattice):
    """Concentric rings (height) divided into angular sectors (width)."""

    name = "polar"

    def __init__(self, sectors: int, rings: int, cell: int):
        self.width, self.height, self.cell = sectors, rings, cell
        self.line_width = max(3, cell // 12)
        self.unit = cell
        self.margin = int(cell * 0.9)
        self.cells = [(x, y) for y in range(rings) for x in range(sectors)]
        self.cellset = set(self.cells)
        self._img = self.margin * 2 + rings * cell * 2
        self.center_px = (self._img / 2, self._img / 2)
        self.outer_r = rings * cell
        self.ring_step = self.outer_r / rings
        self.theta_step = 360 / sectors

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        out = [((x - 1) % self.width, y), ((x + 1) % self.width, y)]
        if y > 0:
            out.append((x, y - 1))
        if y + 1 < self.height:
            out.append((x, y + 1))
        return out

    def is_border(self, cell: Cell) -> bool:
        return cell[1] == self.height - 1

    def center(self, cell: Cell) -> tuple[float, float]:
        x, y = cell
        theta = math.radians((x + 0.5) * self.theta_step - 90)
        radius = (y + 0.5) * self.ring_step
        cx, cy = self.center_px
        return (cx + math.cos(theta) * radius, cy + math.sin(theta) * radius)

    def image_size(self) -> tuple[int, int]:
        return (self._img, self._img)

    def _arc(self, draw, radius, a0, a1, color, line_width):
        cx, cy = self.center_px
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.arc(box, a0, a1, fill=color, width=line_width)

    def draw_walls(self, draw: ImageDraw.ImageDraw, walls: set[Wall],
                   color: tuple, line_width: int) -> None:
        cx, cy = self.center_px
        for y in range(self.height):
            r_in, r_out = y * self.ring_step, (y + 1) * self.ring_step
            for x in range(self.width):
                a1 = math.radians((x + 1) * self.theta_step - 90)
                if wall_key((x, y), ((x + 1) % self.width, y)) in walls:
                    draw.line(
                        (cx + math.cos(a1) * r_in, cy + math.sin(a1) * r_in,
                         cx + math.cos(a1) * r_out, cy + math.sin(a1) * r_out),
                        fill=color, width=line_width,
                    )
                if y + 1 < self.height and wall_key((x, y), (x, y + 1)) in walls:
                    self._arc(draw, r_out, x * self.theta_step - 90,
                              (x + 1) * self.theta_step - 90, color, line_width)
        # The ring-0/ring-1 boundary is ALREADY drawn per cell by the y=0 pass
        # above (its r_out is exactly self.ring_step), and only where `walls`
        # actually holds one. Drawing it here as a full 0-360 circle would paint
        # over those decisions and seal the hub unconditionally, so the picture
        # would show a wall that the gold path walks through.
        #
        # The OUTER boundary keeps its unconditional circle: the outside edge is
        # a wall by construction.
        self._arc(draw, self.outer_r, 0, 360, color, line_width)    # outer boundary


def build_lattice(kind: str, w: int, h: int, cell: int) -> Lattice:
    if kind == "rect":
        return RectLattice(w, h, cell)
    if kind == "hex":
        return HexLattice(w, h, cell)
    if kind == "polar":
        return PolarLattice(w, h, cell)
    raise ValueError(kind)


def distance_px(lat: Lattice, a: Cell, b: Cell) -> float:
    ax, ay = lat.center(a)
    bx, by = lat.center(b)
    return math.hypot(ax - bx, ay - by)



# --------------------------------------------------------------------------- #
# Rendering of a full case (walls + markers)
# --------------------------------------------------------------------------- #
def _marker(draw: ImageDraw.ImageDraw, xy, label: str, color, r: float,
            shape: str = "circle") -> None:
    x, y = xy
    if shape == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=color)
    elif shape == "diamond":
        pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        draw.polygon(pts, fill=color)
    elif shape == "star":
        pts = []
        inner = r * 0.45
        for i in range(10):
            radius = r if i % 2 == 0 else inner
            angle = -math.pi / 2 + i * math.pi / 5
            pts.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        draw.polygon(pts, fill=color)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    if label:
        font = load_bundled_font(int(r * 1.2), "sans-bold")
        bb = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (bb[2] - bb[0]) / 2 - bb[0], y - (bb[3] - bb[1]) / 2 - bb[1]),
                  label, fill=(255, 255, 255), font=font)


def render_case(lat: Lattice, walls: set[Wall], markers: list[tuple],
                style: Style = DEFAULT_STYLE, marker_scale: float = 1.0) -> Image.Image:
    """`marker_scale` enlarges the markers (and their labels, whose font
    tracks the disc radius) without touching the maze geometry. At a fixed output
    resolution the disc is ~0.42*cell, so denser lattices (hex/polar at the harder
    tiers) get tiny labels; callers bump this up per-lattice to keep A/B/C/D
    legible. Individual markers may pass a fifth tuple field as a local scale."""
    lw = lat.styled_width(style.thickness)
    img = Image.new("RGB", lat.image_size(), style.bg)
    draw = ImageDraw.Draw(img)
    lat.draw_walls(draw, walls, style.wall, lw)
    r = (lat.edge_len if isinstance(lat, StraightLattice) else lat.ring_step) * 0.42 * marker_scale
    for marker in markers:
        cell, label, color = marker[0], marker[1], marker[2]
        shape = marker[3] if len(marker) > 3 else "circle"
        local_scale = marker[4] if len(marker) > 4 else 1.0
        _marker(draw, lat.center(cell), label, color, r * local_scale, shape)
    return img


def autocrop(img: Image.Image, bg: tuple, pad: int = 14):
    """Trim background; return cropped image and the (dx, dy) crop origin."""
    diff = ImageChops.difference(img, Image.new("RGB", img.size, bg))
    bbox = diff.getbbox()
    if bbox is None:
        return img, (0, 0)
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(img.width, r + pad), min(img.height, b + pad)
    return img.crop((l, t, r, b)), (l, t)
