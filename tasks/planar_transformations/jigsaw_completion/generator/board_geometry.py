"""Joint jigsaw boards: exact-edge solutions, tile drawing and the upper-tier builder."""

from PIL import ImageDraw
from shared.puzzles.drawing import COLORS, canvas, label, image_size, record


def turn_edges(edges, k):
    return tuple(tuple(edges[(i - k) % 4]) for i in range(4))


def jigsaw_solutions(board, holes, pool, limit=2):
    """Enumerate joint solutions; edge signatures are clockwise around each tile."""
    n = len(board)
    holes = list(map(tuple, holes))
    hole_set = set(holes)
    assignments = {}
    answers = []
    delta = ((-1, 0), (0, 1), (1, 0), (0, -1))
    candidates = [(i, k, turn_edges(p, k)) for i, p in enumerate(pool) for k in range(4)]

    def fits(pos, edges):
        r, c = pos
        for side, (dr, dc) in enumerate(delta):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            if (rr, cc) in assignments:
                other = assignments[rr, cc][2]
            elif (rr, cc) not in hole_set:
                other = board[rr][cc]
            else:
                continue
            if tuple(edges[side]) != tuple(reversed(other[(side + 2) % 4])):
                return False
        return True

    def visit(unused):
        if len(answers) >= limit:
            return
        if len(assignments) == len(holes):
            answers.append([(assignments[p][0], assignments[p][1]) for p in holes])
            return
        branches = []
        for p in holes:
            if p not in assignments:
                opts = [x for x in candidates if x[0] in unused and fits(p, x[2])]
                if not opts:
                    return
                branches.append((len(opts), p, opts))
        _, p, opts = min(branches, key=lambda x: x[0])
        for cand in opts:
            assignments[p] = cand
            visit(unused - {cand[0]})
            del assignments[p]

    visit(set(range(len(pool))))
    return answers


def draw_tile(im, edges, x, y, side):
    d = ImageDraw.Draw(im)
    d.rectangle((x, y, x + side, y + side), fill=(247, 247, 242), outline=(60, 65, 70), width=2)
    ports = len(edges[0])
    thick = max(7, side // 9)
    for k, edge in enumerate(edges):
        for j, color in enumerate(edge):
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
            d.rectangle(box, fill=COLORS[color], outline=(90, 90, 90), width=1)


def jigsaw(tier, rng):
    rank = ('hard', 'xhard', 'xxhard').index(tier)
    n = (3, 4, 5)[rank]
    nh = (2, 3, 5)[rank]
    ports = (2, 3, 4)[rank]
    npool = (6, 8, 10)[rank]
    for attempt in range(500):
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
        start = (rng.randrange(n), rng.randrange(n))
        holes = [start]
        while len(holes) < nh:
            frontier = list(
                {
                    (r + dr, c + dc)
                    for r, c in holes
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < n and 0 <= c + dc < n
                }
                - {*holes}
            )
            holes.append(rng.choice(sorted(frontier)))
        holes.sort()
        pool = [turn_edges(board[r][c], rng.randrange(4)) for r, c in holes]
        for _ in range(npool - nh):
            original = rng.choice(pool[:nh])
            changed = [list(x) for x in original]
            # Shuffle ports on one edge: equal color counts, local near-miss.
            side = rng.choice([i for i, e in enumerate(original) if len(set(e)) > 1] or [0])
            changed[side] = changed[side][1:] + changed[side][:1]
            pool.append(turn_edges(changed, rng.randrange(4)))
        if len({min(turn_edges(p, k) for k in range(4)) for p in pool}) != len(pool):
            continue
        rng.shuffle(pool)
        answers = jigsaw_solutions(board, holes, pool)
        if len(answers) == 1:
            break
    else:
        raise RuntimeError('Could not construct unique joint jigsaw')
    answer = ','.join('ABCDEFGHIJ'[i] + str(k) for i, k in answers[0])
    im = canvas(image_size('jigsaw_completion', tier))
    large = im.width > 448
    size = min(110, 320 // n) if large else min(78, 240 // n)
    ox = (im.width - n * size) // 2
    oy = 18
    for r in range(n):
        for c in range(n):
            x, y = ox + c * size, oy + r * size
            if (r, c) in holes:
                ImageDraw.Draw(im).rectangle((x, y, x + size, y + size), fill=(220, 223, 228), outline='black', width=2)
                label(im, str(holes.index((r, c)) + 1), (x + size / 2 - 5, y + size / 2 - 10), 17)
            else:
                draw_tile(im, board[r][c], x, y, size)
    tile = 70 if large else 54
    columns = 5
    for i, e in enumerate(pool):
        r, c = divmod(i, columns)
        x = 24 + c * (117 if large else 86)
        y = (398 + r * 94) if large else (302 + r * 72)
        label(im, 'ABCDEFGHIJ'[i], (x + 22, y - 15), 12)
        draw_tile(im, e, x, y, tile)
    q = 'Fill every numbered hole using the candidate square pieces. Along each shared edge, the colored strips must continue with exactly matching colors at corresponding positions. Pieces may be rotated by multiples of 90 degrees clockwise, never reflected, and each candidate may be used at most once. Solve all holes together, including edges between holes. Return assignments in hole-number order as comma-separated letter/turn pairs: A0 means candidate A as displayed, A1 means 90 degrees clockwise, A2 means 180, A3 means 270. Example format only: B1,D0.'
    rec = record(
        'jigsaw_completion',
        tier,
        q,
        answer,
        dict(
            family='joint_edge_jigsaw',
            board=board,
            holes=holes,
            pool=pool,
            ports=ports,
            solution=answers[0],
            attempts=attempt + 1,
        ),
    )
    return rec, im
