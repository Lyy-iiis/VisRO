"""Independent verifier for jigsaw_completion.

Imports nothing from the generator (generator/, shared/puzzles) or from the
release adapter.  From the delivered PNG alone it

  1. finds the board (the largest non-white blob) and the candidate squares
     (the other square blobs), splits the board into its n x n grid and tells
     holes (grey, no strips) from visible tiles;
  2. reads every coloured strip of every visible tile and every candidate by
     scanning four pixel rows/columns just inside each edge, so the number of
     strips per edge, their colours and their positions are measured, not
     assumed (strips must be evenly spaced and centred, which is what makes
     "corresponding positions" well defined);
  3. reads the hole numbers and candidate letters by template matching
     against the DejaVuSans glyphs (the committed repo font), so the
     label -> piece mapping the answer string relies on is taken from pixels;
     each letter must sit closer to the candidate below it than to any
     candidate above it ("the letter printed just above it", as the question
     says);
  4. solves the decoded puzzle with its own exhaustive search: every
     assignment of distinct candidates x 4 clockwise quarter-turns to the
     holes, with rotation done by turning strip positions about the tile
     centre and matching done on physical positions (a tile's right strips,
     top to bottom, against its neighbour's left strips, top to bottom), and
     ALL solutions counted -- no early stop;
  5. requires exactly one solution, equal to the gold answer string, and
     requires the record (tier row, question text, answer format, meta board,
     holes, pool, solution, hidden hole tiles) to agree with the pixels;
  6. parses the format illustration out of the question and requires that it
     is not the gold and shares no prefix, letter/turn pair or letter with it,
     and that none of its letters is a real candidate letter of the tier.

The solution PNG is not an input here (verify() gets the question image only).
"""
from __future__ import annotations

import os
import re
from fractions import Fraction
from functools import lru_cache

# Generation runs one process per worker; OpenBLAS (pulled in by numpy) would otherwise
# start one busy-waiting thread per core in every worker.  Set before numpy is first
# imported; an explicit environment setting wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from visro.paths import ROOT  # noqa: E402

CORRECTNESS = ("The board grid, holes, every coloured strip of every visible tile and candidate, the hole numbers "
               "and the candidate letters are decoded from the PNG (strip scan + glyph template match), the decoded "
               "puzzle is solved by an exhaustive geometric-rotation search written for this verifier, and the single "
               "solution must equal the gold string; meta (board, holes, pool, solution, hidden tiles) must equal "
               "the decoding.")
UNIQUENESS = ("The search enumerates every assignment of distinct candidates and all four quarter-turns to the holes "
              "with no early stop (so symmetric pieces, boundary-only differences and hole-hole edges are covered) "
              "and rejects the item unless exactly one assignment satisfies every shared edge; candidates must also "
              "be pairwise distinct up to rotation so a letter is never ambiguous.")

QUESTION = (
    'Fill each numbered gray hole with a lettered piece so colored strips continue across every '
    'shared edge, including between holes; board-border sides are free. Pieces turn clockwise in '
    'quarter turns, never flip, and are used at most once. Answer letter+turns (0-3, from as '
    'drawn) per hole in hole order, comma-separated, e.g. X1,Y0,Z3.'
)
EXAMPLE_RE = re.compile(r"e\.g\. ([^ ]+)\.$")

TIER = {  # board side n, holes, strips per edge, candidates, canvas px
    "easy": (3, 2, 2, 4, 448),
    "medium": (3, 2, 2, 5, 448),
    "hard": (3, 2, 2, 6, 448),
    "xhard": (4, 3, 3, 8, 588),
    "xxhard": (5, 5, 4, 10, 588),
}
PALETTE = {(199, 49, 58): 1, (42, 103, 193): 2, (40, 151, 99): 3, (231, 182, 37): 4, (142, 78, 172): 5}
TILE_FILL = (247, 247, 242)
HOLE_FILL = (220, 223, 228)
STRIP_LINE = (90, 90, 90)
LETTERS = "ABCDEFGHIJ"
FONT = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
SIDES = ("top", "right", "bottom", "left")


class Bad(Exception):
    pass


def need(ok, msg):
    if not ok:
        raise Bad(msg)


# --------------------------------------------------------------------------- image decode
def _runs(line):
    """Runs of palette colour along a 1-px scan line -> [(colour, start, end)] (end exclusive).

    Anything other than tile fill, strip outline or a palette colour on the
    scan line means a strip is overpainted or the tile is not a strip tile."""
    out = []
    cur = None
    for i, px in enumerate(map(tuple, line)):
        if px in PALETTE:
            col = PALETTE[px]
            if cur is not None and cur[0] == col and cur[2] == i:
                cur[2] = i + 1
            else:
                if cur is not None:
                    out.append(tuple(cur))
                cur = [col, i, i + 1]
        else:
            need(px in (TILE_FILL, STRIP_LINE), f"unexpected colour {px} on a strip scan line")
            if cur is not None:
                out.append(tuple(cur))
                cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def _edge(a, x, y, side, which, ports):
    """Colours along one edge in reading order (top/bottom: left->right, left/right: top->bottom)."""
    lo, hi = 2, side - 1          # inside the 2-px tile outline
    if which == "top":
        lines = [a[y + d, x + lo:x + hi] for d in (3, 4, 5, 6)]
    elif which == "bottom":
        lines = [a[y + side - d, x + lo:x + hi] for d in (3, 4, 5, 6)]
    elif which == "left":
        lines = [a[y + lo:y + hi, x + d] for d in (3, 4, 5, 6)]
    else:
        lines = [a[y + lo:y + hi, x + side - d] for d in (3, 4, 5, 6)]
    runs = [_runs(line) for line in lines]
    need(all(r == runs[0] for r in runs), f"{which} strips not uniform across their depth at tile ({x},{y})")
    runs = runs[0]
    need(len(runs) == ports, f"{which} edge of tile ({x},{y}) shows {len(runs)} strips, tier has {ports}")
    need(all(e - s >= 3 for _, s, e in runs), f"a {which} strip of tile ({x},{y}) is under 3 px")
    centres = [(s + e - 1) / 2 + lo for _, s, e in runs]
    mid = side / 2
    for j in range(ports):          # centred and evenly spaced -> positions correspond across tiles
        need(abs(centres[j] + centres[ports - 1 - j] - 2 * mid) <= 1.5, f"strips of tile ({x},{y}) not centred")
        want = mid + (centres[-1] - centres[0]) * (j / (ports - 1) - 0.5) if ports > 1 else mid
        need(abs(centres[j] - want) <= 1.5, f"strips of tile ({x},{y}) not evenly spaced")
    return tuple(c for c, _, _ in runs)


def _tile(a, x, y, side, ports):
    return {w: _edge(a, x, y, side, w, ports) for w in SIDES}


@lru_cache(None)
def _templates(chars):
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for size in range(10, 21):
        f = ImageFont.truetype(str(FONT), size)
        for ch in chars:
            for sub in (0.0, 0.5):
                im = Image.new("L", (40, 40), 255)
                ImageDraw.Draw(im).text((5 + sub, 5 + sub), ch, fill=0, font=f)
                g = np.asarray(im) < 140
                ys, xs = np.nonzero(g)
                out.append((ch, size, g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out


def _iou(g, t):
    best = 0.0
    if abs(g.shape[0] - t.shape[0]) > 1 or abs(g.shape[1] - t.shape[1]) > 1:
        return 0.0
    H, W = max(g.shape[0], t.shape[0]) + 2, max(g.shape[1], t.shape[1]) + 2
    G = np.zeros((H, W), bool)
    G[1:1 + g.shape[0], 1:1 + g.shape[1]] = g
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            T = np.zeros((H, W), bool)
            if dy + t.shape[0] > H or dx + t.shape[1] > W:
                continue
            T[dy:dy + t.shape[0], dx:dx + t.shape[1]] = t
            u = (G | T).sum()
            best = max(best, (G & T).sum() / u if u else 0.0)
    return best


def _read_glyph(mask, chars):
    """Classify one dark-pixel glyph mask; returns (char, margin)."""
    ys, xs = np.nonzero(mask)
    need(len(ys) >= 6, "label glyph missing or too small")
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    score = {}
    for ch, _, t in _templates(chars):
        score[ch] = max(score.get(ch, 0.0), _iou(g, t))
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    (c1, s1), (_, s2) = ranked[0], ranked[1]
    need(s1 >= 0.8 and s1 - s2 >= 0.12, f"label glyph not legible (best {c1} {s1:.2f}, runner-up {s2:.2f})")
    return c1


def decode(image, tier):
    """PNG -> {'n', 'size', 'origin', 'holes': {number: (r, c)}, 'board': {(r, c): tile},
    'pool': {letter: tile}, 'boxes': {letter: (x, y, side)}, 'label_gaps': {letter: (px to own candidate,
    px to the nearest candidate above or None)}}; a tile is {side name: colours in reading order}."""
    from scipy import ndimage

    n, nh, ports, npool, canvas = TIER[tier]
    a = np.asarray(image.convert("RGB"))
    need(a.shape[:2] == (canvas, canvas), f"image is {a.shape[1]}x{a.shape[0]}, tier canvas is {canvas}")
    nonwhite = (a != 255).any(axis=2)
    lab, k = ndimage.label(nonwhite, structure=np.ones((3, 3), int))
    objs = ndimage.find_objects(lab)
    blobs = []
    for i, sl in enumerate(objs, 1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
        blobs.append((i, x0, y0, x1, y1, int((lab[sl] == i).sum())))
    big = [b for b in blobs if b[3] - b[1] >= 30 and b[4] - b[2] >= 30]
    small = [b for b in blobs if not (b[3] - b[1] >= 30 and b[4] - b[2] >= 30)]
    need(len(big) == npool + 1, f"found {len(big) - 1} candidate squares, tier has {npool}")
    board = max(big, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
    _, bx0, by0, bx1, by1, _ = board
    need(bx1 - bx0 == by1 - by0 and (bx1 - bx0) % n == 0, "board is not an n x n grid of equal squares")
    size = (bx1 - bx0) // n
    for q in range(n + 1):          # every grid line is drawn
        col = a[by0:by1 + 1, bx0 + q * size]
        row = a[by0 + q * size, bx0:bx1 + 1]
        dark = lambda v: (v.astype(int).sum(axis=1) < 300).mean()
        need(dark(col) > 0.9 and dark(row) > 0.9, f"board grid line {q} missing (n={n})")
    cands = [b for b in big if b is not board]
    for _, x0, y0, x1, y1, _ in cands:
        need(x1 - x0 == y1 - y0, "a candidate is not square")
        need(y0 > by1, "a candidate overlaps the board rows")

    # board cells
    holes_rc, tiles = [], {}
    for r in range(n):
        for c in range(n):
            x, y = bx0 + c * size, by0 + r * size
            corner = tuple(a[y + 4, x + 4])
            if corner == HOLE_FILL:
                inner = a[y + 3:y + size - 2, x + 3:x + size - 2].reshape(-1, 3)
                need(not any(tuple(p) in PALETTE for p in np.unique(inner, axis=0)), f"hole ({r},{c}) shows strips")
                need((inner == HOLE_FILL).all(axis=1).mean() > 0.85, f"hole ({r},{c}) is not drawn empty")
                holes_rc.append((r, c))
            else:
                need(corner == TILE_FILL, f"board cell ({r},{c}) is neither a tile nor a hole")
                tiles[(r, c)] = _tile(a, x, y, size, ports)
    need(len(holes_rc) == nh, f"board shows {len(holes_rc)} holes, tier has {nh}")

    # hole numbers (dark glyph inside the grey cell)
    holes = {}
    for (r, c) in holes_rc:
        x, y = bx0 + c * size, by0 + r * size
        sub = a[y + 3:y + size - 2, x + 3:x + size - 2].astype(int).sum(axis=2) < 3 * 140
        digit = _read_glyph(sub, "123456789")
        need(digit not in holes, f"hole number {digit} appears twice")
        holes[int(digit)] = (r, c)
    need(sorted(holes) == list(range(1, nh + 1)), f"hole numbers {sorted(holes)} are not 1..{nh}")

    # candidate letters: the glyph blobs directly above each candidate square
    pool, boxes, gaps = {}, {}, {}
    used = set()
    for _, x0, y0, x1, y1, _ in cands:
        mine = [b for b in small if y0 - 26 <= b[4] < y0 and x0 - 4 <= (b[1] + b[3]) / 2 <= x1 + 4]
        need(mine, f"candidate at ({x0},{y0}) has no letter")
        used.update(b[0] for b in mine)
        gx0, gy0 = min(b[1] for b in mine), min(b[2] for b in mine)
        gx1, gy1 = max(b[3] for b in mine) + 1, max(b[4] for b in mine) + 1
        # "the letter printed just above it": the glyph must sit closer to this
        # candidate's top than to the bottom of any candidate it is printed under
        below = y0 - gy1
        above = [gy0 - (oy1 + 1) for _, ox0, _, ox1, oy1, _ in cands
                 if oy1 < gy0 and ox0 <= (gx0 + gx1) / 2 <= ox1]
        need(not above or below < min(above),
             f"letter above candidate at ({x0},{y0}) is {below} px from it but {min(above or [0])} px from the "
             "candidate above")
        mask = np.isin(lab[gy0:gy1, gx0:gx1], [b[0] for b in mine])
        mask &= a[gy0:gy1, gx0:gx1].astype(int).sum(axis=2) < 3 * 140
        letter = _read_glyph(mask, LETTERS[:npool])
        need(letter not in pool, f"letter {letter} labels two candidates")
        gaps[letter] = (below, min(above) if above else None)
        pool[letter] = _tile(a, x0, y0, x1 - x0, ports)
        boxes[letter] = (x0, y0, x1 - x0)
    need(sorted(pool) == list(LETTERS[:npool]), f"candidate letters {sorted(pool)}")
    stray = [b for b in small if b[0] not in used]
    need(not stray, f"{len(stray)} unexplained marks outside the board and candidates")
    return {"n": n, "size": size, "origin": (bx0, by0), "holes": holes, "board": tiles, "pool": pool,
            "boxes": boxes, "label_gaps": gaps}


# --------------------------------------------------------------------------- geometry
def _points(tile):
    """Physical strip points: {(x, y): colour} on the unit square centred at 0 (y down)."""
    p = len(tile["top"])
    t = [Fraction(2 * j + 1 - p, 2 * p) for j in range(p)]
    h = Fraction(1, 2)
    pts = {}
    for j in range(p):
        pts[(t[j], -h)] = tile["top"][j]
        pts[(h, t[j])] = tile["right"][j]
        pts[(t[j], h)] = tile["bottom"][j]
        pts[(-h, t[j])] = tile["left"][j]
    return pts


def _from_points(pts, p):
    t = [Fraction(2 * j + 1 - p, 2 * p) for j in range(p)]
    h = Fraction(1, 2)
    return {"top": tuple(pts[(t[j], -h)] for j in range(p)), "right": tuple(pts[(h, t[j])] for j in range(p)),
            "bottom": tuple(pts[(t[j], h)] for j in range(p)), "left": tuple(pts[(-h, t[j])] for j in range(p))}


def rotate(tile, k):
    """Turn a tile k quarter-turns clockwise on screen: (x, y) -> (-y, x) with y pointing down."""
    pts = _points(tile)
    for _ in range(k % 4):
        pts = {(-y, x): v for (x, y), v in pts.items()}
    return _from_points(pts, len(tile["top"]))


def _key(tile):
    return tuple(tile[s] for s in SIDES)


def fits(cell, tile, n, known):
    """`known` maps (r, c) -> tile for visible and already-filled cells."""
    r, c = cell
    pairs = (((r - 1, c), "top", "bottom"), ((r, c + 1), "right", "left"),
             ((r + 1, c), "bottom", "top"), ((r, c - 1), "left", "right"))
    for nb, mine, theirs in pairs:
        other = known.get(nb)
        if other is not None and tile[mine] != other[theirs]:
            return False
    return True


def solve_all(n, board, holes, pool):
    """Every assignment hole-number -> (letter, turn) satisfying all shared edges."""
    order = [holes[i] for i in sorted(holes)]
    letters = sorted(pool)
    turned = {L: [rotate(pool[L], k) for k in range(4)] for L in letters}
    known = dict(board)
    sols = []
    pick = []

    def go(i, used):
        if i == len(order):
            sols.append(tuple(pick))
            return
        cell = order[i]
        for L in letters:
            if L in used:
                continue
            for k in range(4):
                t = turned[L][k]
                if fits(cell, t, n, known):
                    known[cell] = t
                    pick.append((L, k))
                    go(i + 1, used | {L})
                    pick.pop()
                    del known[cell]

    go(0, frozenset())
    return sols


def _cw(tile):
    """Physical reading order -> the record's clockwise edge encoding."""
    return [list(tile["top"]), list(tile["right"]), list(tile["bottom"][::-1]), list(tile["left"][::-1])]


def _phys(cw):
    return {"top": tuple(cw[0]), "right": tuple(cw[1]), "bottom": tuple(cw[2][::-1]), "left": tuple(cw[3][::-1])}


# --------------------------------------------------------------------------- format example
def example_problems(question, answer, npool=len(LETTERS)):
    """The question's format illustration must not be, or share any part of, the gold.

    Parsed from the question text itself, so it holds for whatever example the
    question carries: not equal, no common prefix, no common letter/turn pair
    (in any position), no common letter, and every example letter outside the
    tier's candidate letters (so the example can never be a valid answer)."""
    m = EXAMPLE_RE.search(question or "")
    if m is None:
        return ["question has no parsable format illustration"]
    answer = answer or ""
    ex, gold = m.group(1).split(","), answer.split(",")
    out = []
    if ex == gold:
        out.append("answer equals the question's format example (copyable)")
    if ex[0] == gold[0] or m.group(1).startswith(answer) or answer.startswith(m.group(1)):
        out.append("answer shares a prefix with the question's format example")
    common = sorted(set(ex) & set(gold))
    if common:
        out.append(f"answer shares pair(s) {common} with the question's format example")
    letters = sorted({p[:1] for p in ex} & {p[:1] for p in gold})
    if letters:
        out.append(f"answer shares letter(s) {letters} with the question's format example")
    valid = [p for p in ex if p[:1] in LETTERS[:npool]]
    if valid:
        out.append(f"format example pair(s) {valid} use real candidate letters")
    return out


# --------------------------------------------------------------------------- verify
def _verify(rec, image):
    tier = rec.get("tier")
    need(tier in TIER, f"unknown tier {tier!r}")
    n, nh, ports, npool, canvas = TIER[tier]
    need(rec.get("task") == "jigsaw_completion", "wrong task")
    need(rec.get("question") == QUESTION, "question text is not the declared joint-jigsaw form")
    need(rec.get("answer_type") == "token-list" and rec.get("choices") is None, "answer_type must be token-list")
    ans = rec.get("answer", "")
    hi = LETTERS[npool - 1]
    need(re.fullmatch(rf"[A-{hi}][0-3](?:,[A-{hi}][0-3]){{{nh - 1}}}", ans or "") is not None,
         f"answer {ans!r} is not {nh} comma-separated <letter A-{hi}><turn 0-3> pairs")
    gold = tuple((p[0], int(p[1])) for p in ans.split(","))
    need(len({L for L, _ in gold}) == nh, "answer uses a candidate twice")
    ex = example_problems(rec["question"], ans, npool)
    need(not ex, "; ".join(ex))

    d = decode(image, tier)
    sols = solve_all(n, d["board"], d["holes"], d["pool"])
    need(len(sols) == 1, f"{len(sols)} valid assignments for the decoded puzzle (need exactly 1)")
    need(sols[0] == gold, f"decoded puzzle solves to {','.join(L + str(k) for L, k in sols[0])}, gold is {ans}")

    classes = {min(_key(rotate(t, k)) for k in range(4)) for t in d["pool"].values()}
    need(len(classes) == npool, "two candidates are equal up to rotation")
    for (r, c), t in d["board"].items():   # the visible board itself obeys the rule it teaches
        for nb, mine, theirs in (((r, c + 1), "right", "left"), ((r + 1, c), "bottom", "top")):
            if nb in d["board"]:
                need(t[mine] == d["board"][nb][theirs], f"visible tiles {r, c} and {nb} disagree")

    # the record must say what the picture says
    m = rec.get("meta") or {}
    need(m.get("n") == n and m.get("ports") == ports, "meta n/ports differ from the tier row")
    holes_meta = [tuple(h) for h in m.get("holes", [])]
    need(holes_meta == [d["holes"][i] for i in range(1, nh + 1)], "meta holes differ from the numbered holes drawn")
    need(len(m.get("board", [])) == n and all(len(row) == n for row in m["board"]), "meta board size")
    for (r, c), t in d["board"].items():
        need(m["board"][r][c] == _cw(t), f"meta board tile ({r},{c}) differs from the image")
    need(len(m.get("pool", [])) == npool, "meta pool size")
    for i, L in enumerate(LETTERS[:npool]):
        need(m["pool"][i] == _cw(d["pool"][L]), f"meta candidate {L} differs from the image")
    need([list(s) for s in m.get("solution", [])] == [[LETTERS.index(L), k] for L, k in gold],
         "meta solution differs from the answer")
    for i, (L, k) in enumerate(gold, 1):   # hidden tile == placed candidate
        r, c = d["holes"][i]
        need(_phys(m["board"][r][c]) == rotate(d["pool"][L], k), f"hidden tile of hole {i} is not {L}{k}")


def verify(record: dict, image) -> list[str]:
    try:
        _verify(record, image)
    except Bad as exc:
        return [str(exc)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return [f"malformed record: {type(exc).__name__}: {exc}"]
    return []
