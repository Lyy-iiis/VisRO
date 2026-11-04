"""easy and medium tiers of the `joint_edge_jigsaw` family.

The upper tiers (board_geometry.py jigsaw()) dig several connected holes and ask
for a simultaneous assignment in which the holes also have to agree with EACH
OTHER.  This module builds easy and medium as the same problem with looser
parameters: it asks the upper tiers' question byte for byte (see
`parent_question()`), publishes `meta['family']=='joint_edge_jigsaw'`, sets
`rec['builder_family']` so `shared/puzzles/oracles.py` applies unchanged, and
answers with the same comma-separated letter/turn string.

THE LADDER
----------
    tier    board  holes  ports  pool  decoy
    easy    3x3    2      2      4     recolour
    medium  3x3    2      2      5     recolour
    hard    3x3    2      2      6     port swap      <- board_geometry.py
    xhard   4x4    3      3      8     port swap      <- board_geometry.py
    xxhard  5x5    5      4     10     port swap      <- board_geometry.py

No axis exceeds hard (board_geometry.py: `n=(3,4,5)[rank]; nh=(2,3,5)[rank];
ports=(2,3,4)[rank]; npool=(6,8,10)[rank]`).

  * easy and medium keep hard's 3x3 board and offer a smaller pool.  A 2x2
    board would leave each hole one visible neighbour, so more candidate pairs
    would survive the visible-edge filter than at hard.
  * the decoy axis: hard builds every distractor by cyclically shifting the
    ports of one edge, which for ports=2 swaps two colours and preserves the
    edge's colour multiset -- a near-miss that survives any counting shortcut.
    easy/medium instead REPLACE one port colour, so a distractor differs from
    its parent tile in a colour that is simply not there.
  * rotations are 0/90/180/270 at every tier, as at hard, so every turn digit
    of the answer stays live.

CORE KEY
--------
`adapter.core_key` reads `m['holes']`, blanks those cells of `m['board']` and
appends `m.get('allowed_rotations',[0,1,2,3])`.  meta here publishes `holes`,
`board` and an explicit `allowed_rotations=[0,1,2,3]` (the value hard gets
from the default), so easy, medium and hard produce cores of one shape.

Generation uses `board_geometry.jigsaw_solutions`; validation uses
`oracles.independent_jigsaw`, an independently written solver.
"""
import ast, re
from functools import lru_cache
from PIL import ImageDraw
from pathlib import Path
from shared.puzzles import drawing as draw, oracles
from . import board_geometry as geometry
from shared.puzzles.runtime import record, rng_for, plain

FAMILY = 'joint_edge_jigsaw'
LETTERS = 'ABCDEFGHIJ'
DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))
ATTEMPTS = 4000
HOLE_FILL = (220, 223, 228)

# n     -- board side
# holes -- connected cells removed, filled jointly
# ports -- coloured strips per tile edge
# pool  -- candidate squares offered
# decoy -- 'recolour' replaces one port colour; 'swap' is hard's port shift
# easy uses a 3x3 board, not 2x2: on 2x2 each hole has one visible neighbour, and
# more candidate pairs survive the visible-edge filter than at hard.  With hard's
# board and a smaller pool, every axis stays at or below hard.
CONFIG = {
    'easy':   dict(n=3, holes=2, ports=2, pool=4, decoy='recolour'),
    'medium': dict(n=3, holes=2, ports=2, pool=5, decoy='recolour'),
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def requirements(tier):
    """CONFIG as meta stores it (shared.record runs plain() over meta)."""
    return plain(CONFIG[tier])


# --------------------------------------------------------------------------- #
# The question: byte-identical to the upper tiers'.
# --------------------------------------------------------------------------- #
QUESTION = ('Fill every numbered hole using the candidate square pieces. Along each shared edge, the colored strips must '
            'continue with exactly matching colors at corresponding positions. Pieces may be rotated by multiples of 90 '
            'degrees clockwise, never reflected, and each candidate may be used at most once. Solve all holes together, '
            'including edges between holes. Return assignments in hole-number order as comma-separated letter/turn pairs: '
            'A0 means candidate A as displayed, A1 means 90 degrees clockwise, A2 means 180, A3 means 270. Example format '
            'only: B1,D0.')


@lru_cache(None)
def parent_question():
    """The literal assigned to `q` inside board_geometry.py::jigsaw, read by AST.

    `check()` uses it to prove that the low-tier wording is the upper tiers'.
    """
    tree = ast.parse(Path(geometry.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'jigsaw')
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'q' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('Could not locate the hard-tier jigsaw question')


def question(meta):
    """Signature is question(meta) for every low-tier module; the joint
    wording happens to be parameter-free, exactly as it is at hard."""
    return QUESTION


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def consistent_board(rng, n, ports):
    """Hard's own construction: random tiles, then glue every shared edge."""
    edge = lambda: tuple(rng.randrange(1, 6) for _ in range(ports))
    board = [[tuple(edge() for _ in range(4)) for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            e = list(board[r][c])
            if r:
                e[0] = tuple(reversed(board[r - 1][c][2]))
            if c:
                e[3] = tuple(reversed(board[r][c - 1][1]))
            board[r][c] = tuple(e)
    return board


def grow_holes(rng, n, count):
    """Hard's own growth: a connected cell set, so the holes constrain each other."""
    holes = [(rng.randrange(n), rng.randrange(n))]
    while len(holes) < count:
        frontier = sorted({(r + dr, c + dc) for r, c in holes for dr, dc in DELTA
                           if 0 <= r + dr < n and 0 <= c + dc < n} - {*holes})
        holes.append(rng.choice(frontier))
    return sorted(holes)


def decoy_tile(rng, board, n, hole, mode, ports):
    """A near-miss built from the true tile of `hole`.

    The altered side is always one that faces another cell, never the outer
    boundary: a boundary edge is unconstrained, so altering it would leave a
    second legal solution and the uniqueness filter would just throw the
    candidate away.
    """
    r, c = hole
    original = [list(x) for x in board[r][c]]
    sides = [s for s, (dr, dc) in enumerate(DELTA) if 0 <= r + dr < n and 0 <= c + dc < n]
    if mode == 'swap':
        sides = [s for s in sides if len(set(original[s])) > 1] or sides
    side = rng.choice(sides)
    if mode == 'recolour':
        port = rng.randrange(ports)
        original[side][port] = rng.choice([v for v in range(1, 6) if v != original[side][port]])
    else:
        original[side] = original[side][1:] + original[side][:1]
    return tuple(tuple(x) for x in original)


def canonical_tile(tile):
    return min(geometry.turn_edges(tile, k) for k in range(4))


def build(tier, seed, index):
    cfg = CONFIG[tier]
    n, nh, ports, npool = cfg['n'], cfg['holes'], cfg['ports'], cfg['pool']
    rng = rng_for('jigsaw-joint-low', tier, seed, index)
    for attempt in range(ATTEMPTS):
        board = consistent_board(rng, n, ports)
        holes = grow_holes(rng, n, nh)
        pool = [geometry.turn_edges(board[r][c], rng.randrange(4)) for r, c in holes]
        # Decoy parents are dealt round-robin (hard draws them independently):
        # with independent draws many true tiles would have no near-twin, and
        # "the candidate that nothing else resembles is a true tile" would be a
        # free half-answer.  Round-robin gives every true tile at least one
        # look-alike; the rotating start keeps the extra decoy off hole 1.
        start = rng.randrange(nh)
        for j in range(npool - nh):
            changed = decoy_tile(rng, board, n, holes[(start + j) % nh], cfg['decoy'], ports)
            pool.append(geometry.turn_edges(changed, rng.randrange(4)))
        # Two candidates equal up to rotation would make the answer letter ambiguous.
        if len({canonical_tile(p) for p in pool}) != len(pool):
            continue
        rng.shuffle(pool)
        answers = geometry.jigsaw_solutions(board, holes, pool)
        if len(answers) != 1:
            continue
        answer = ','.join(LETTERS[i] + str(k) for i, k in answers[0])
        meta = dict(family=FAMILY, board=board, holes=holes, pool=pool, ports=ports,
                    solution=answers[0], allowed_rotations=[0, 1, 2, 3],
                    requirements=requirements(tier), attempts=attempt + 1)
        rec = record('jigsaw_completion', tier, seed, index, question(meta), answer, meta, 'exact')
        rec['builder_family'] = FAMILY
        check(rec)
        return rec
    raise RuntimeError('Could not construct a uniquely solved low-tier joint jigsaw')


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
BOARD_TOP = 20
POOL_TOP = 338


def tile_strips(x, y, side, edges):
    """Every coloured strip draw_tile() paints, as (colour, centre pixel).

    Mirrors board_geometry.py::draw_tile exactly; used only to verify the image
    against the metadata, never to draw.
    """
    ports = len(edges[0])
    thick = max(7, side // 9)
    out = []
    for k, edge in enumerate(edges):
        for j, colour in enumerate(edge):
            a = 8 + (side - 16) * j / ports
            b = 8 + (side - 16) * (j + 1) / ports
            if k == 0:
                box = (x + a, y + 2, x + b, y + thick)
            elif k == 1:
                box = (x + side - thick, y + a, x + side - 2, y + b)
            elif k == 2:
                box = (x + side - b, y + side - thick, x + side - a, y + side - 2)
            else:
                box = (x + 2, y + side - b, x + thick, y + side - a)
            out.append((int(colour), (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))))
    return out


def layout(rec):
    """Where every tile of this record sits.  Published on the record so the
    pixel check needs nothing but the record and the image."""
    m = rec['meta']
    n = len(m['board'])
    size = 112 if n == 2 else 86
    ox = (draw.SIZE - n * size) // 2
    holes = [tuple(h) for h in m['holes']]
    tiles = []
    for r in range(n):
        for c in range(n):
            x, y = ox + c * size, BOARD_TOP + r * size
            if (r, c) in holes:
                tiles.append(dict(name=f'hole-{holes.index((r, c)) + 1}', x=x, y=y, side=size, edges=None))
            else:
                tiles.append(dict(name=f'board-{r}-{c}', x=x, y=y, side=size, edges=plain(m['board'][r][c])))
    count = len(m['pool'])
    side = 72 if count == 4 else 64
    stride = draw.SIZE / count
    for i, p in enumerate(m['pool']):
        x = round(i * stride + (stride - side) / 2)
        tiles.append(dict(name=f'candidate-{LETTERS[i]}', x=x, y=POOL_TOP, side=side, edges=plain(p)))
    return tiles


def paint(rec):
    im = draw.canvas(draw.image_size('jigsaw_completion', rec['difficulty']))
    for spot in layout(rec):
        if spot['edges'] is not None:
            geometry.draw_tile(im, spot['edges'], spot['x'], spot['y'], spot['side'])
            if spot['name'].startswith('candidate-'):
                draw.label(im, spot['name'][-1], (spot['x'] + spot['side'] / 2 - 5, spot['y'] - 21), 15)
        else:
            number = spot['name'].split('-')[1]
            ImageDraw.Draw(im).rectangle((spot['x'], spot['y'], spot['x'] + spot['side'], spot['y'] + spot['side']),
                                         fill=HOLE_FILL, outline='black', width=2)
            draw.label(im, number, (spot['x'] + spot['side'] / 2 - 5, spot['y'] + spot['side'] / 2 - 10), 18)
    return im


def render(rec):
    im = paint(rec)
    rec['image_labels'] = im.info.get('diagram_labels', [])
    rec['render_tiles'] = layout(rec)
    return im


# --------------------------------------------------------------------------- #
# Independent checks
# --------------------------------------------------------------------------- #
def check(rec):
    """Structural audit plus the independent joint solver.

    Generation calls `geometry.jigsaw_solutions`; validation calls
    `oracles.independent_jigsaw` (a different algorithm in a different fixed
    file, working on a rasterised tile), so neither can rubber-stamp itself.
    """
    m = rec['meta']
    tier = rec['difficulty']
    require(tier in CONFIG, 'Wrong joint-jigsaw tier')
    cfg = CONFIG[tier]
    require(m.get('family') == FAMILY and rec.get('builder_family') == FAMILY, 'Not the joint edge-jigsaw family')
    require(rec['question'] == question(m), 'The public joint-jigsaw contract changed')
    require(QUESTION == parent_question(), 'Published wording is no longer the hard-tier wording')
    require(m.get('requirements') == requirements(tier), 'Changed joint-jigsaw tier parameters')
    require(rec['ans_type'] == 'exact' and not rec['options'], 'Joint jigsaw is not a multiple choice item')
    n = len(m['board'])
    require(n == cfg['n'] and all(len(row) == n for row in m['board']), 'Changed board size')
    require(len(m['pool']) == cfg['pool'] and m['ports'] == cfg['ports'], 'Changed pool or port count')
    require(m['allowed_rotations'] == [0, 1, 2, 3], 'Changed allowed rotations')
    for tile in [t for row in m['board'] for t in row] + [t for t in m['pool']]:
        require(len(tile) == 4 and all(len(e) == cfg['ports'] and all(type(v) is int and 1 <= v <= 5 for v in e)
                                       for e in tile), 'Bad edge strips')
    holes = [tuple(h) for h in m['holes']]
    require(len(holes) == cfg['holes'] == len(set(holes)), 'Wrong or repeated holes')
    require(all(len(h) == 2 and all(type(v) is int and 0 <= v < n for v in h) for h in holes), 'Hole outside board')
    require(holes == sorted(holes), 'Holes are not in canonical reading order')
    reached = {holes[0]}
    while True:
        grown = reached | {h for h in holes for r, c in reached if abs(h[0] - r) + abs(h[1] - c) == 1}
        if grown == reached:
            break
        reached = grown
    require(reached == set(holes), 'Holes are not one connected region')
    for r in range(n):
        for c in range(n):
            if (r, c) in holes:
                continue
            for side, (rr, cc) in ((1, (r, c + 1)), (2, (r + 1, c))):
                if rr < n and cc < n and (rr, cc) not in holes:
                    require(list(m['board'][r][c][side]) == list(reversed(m['board'][rr][cc][(side + 2) % 4])),
                            'Visible pieces have inconsistent shared edges')
    require(len({canonical_tile(tuple(map(tuple, p))) for p in m['pool']}) == len(m['pool']),
            'Two candidates coincide up to rotation')
    require(re.fullmatch('[A-%s][0-3](?:,[A-%s][0-3])*' % (LETTERS[cfg['pool'] - 1], LETTERS[cfg['pool'] - 1]),
                         rec['answer']), 'Malformed joint assignment')
    used = [pair[0] for pair in rec['answer'].split(',')]
    require(len(rec['answer'].split(',')) == cfg['holes'] and len(set(used)) == cfg['holes'],
            'A candidate is used twice')
    require(rec['answer'] == ','.join(LETTERS[i] + str(k) for i, k in m['solution']), 'Answer does not match meta')
    # The hidden tile of each hole must BE the answered candidate under the answered
    # turn.  Nothing else looks at those cells (core_key blanks them, the fixed
    # solver ignores them), so this is the only check that meta's board is consistent.
    for (r, c), (idx, turn) in zip(holes, m['solution']):
        placed = geometry.turn_edges(tuple(map(tuple, m['pool'][idx])), turn)
        require(placed == tuple(map(tuple, m['board'][r][c])), 'Hidden hole tile is not the answered candidate')
    detail = oracles.check(rec)         # uniqueness + gold, by the fixed validator
    return dict(detail, holes=len(holes), candidates=len(m['pool']),
                placements=len(m['pool']) * 4, unique_answer=True)


def check_render(rec, im):
    """Pixel audit: every coloured strip in the image equals the metadata.

    `oracles.check_render` has no jigsaw branch, so this is where the picture and
    the metadata are compared.
    """
    size = draw.image_size('jigsaw_completion', rec['difficulty'])
    require(im.mode == 'RGB' and im.size == (size, size), 'Image profile mismatch')
    tiles = rec['render_tiles']
    require(tiles == layout(rec), 'Published layout does not match the metadata')
    seen = set()
    for spot in tiles:
        x, y, side = spot['x'], spot['y'], spot['side']
        require(0 <= x and 0 <= y and x + side < im.width and y + side < im.height, 'Tile outside the canvas')
        key = (x, y)
        require(key not in seen, 'Two tiles share an origin')
        seen.add(key)
        if spot['edges'] is None:
            require(im.getpixel((x + 5, y + 5)) == HOLE_FILL, 'Hole is not drawn empty')
            continue
        for colour, point in tile_strips(x, y, side, spot['edges']):
            require(im.getpixel(point) == draw.COLORS[colour], f'Strip pixel differs from metadata at {point}')
    require(sorted(rec['image_labels']) == sorted([LETTERS[i] for i in range(len(rec['meta']['pool']))] +
                                                  [str(i + 1) for i in range(len(rec['meta']['holes']))]),
            'Unexpected text in the question image')
    return dict(tiles=len(tiles), strips=sum(len(t['edges'][0]) * 4 for t in tiles if t['edges']))
