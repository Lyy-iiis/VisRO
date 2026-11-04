"""metro_map boards: the line grower, the chord search, route statistics and station names.

Stations are cells of a GRID x GRID grid; a line is an octilinear polyline
through stations, and a cell on two or more lines is an interchange.

THE SCALAR.  kappa = mean over all simple A->B routes (cutoff L*+14 edges) of
(stops on that route) - (stops on the minimum route).  kappa = 0 means every
route from A to B has the same stop count, so the word "minimum" in the
question does no work; the tiers draw their min_stops pairs from increasing
kappa strata (metro_core).

CHORDS.  Cycles are built in, so alternative A-B routes exist with DIFFERENT
stop counts.  A chord runs from an existing station X to an existing station Y
with FEWER intermediate stations than the route that already joins them (delta
in CHORD_DELTA).  The sign matters: a chord is a NEW line, so riding it costs
two transfers, and a longer chord would never shorten a route.

NO PHANTOM STATIONS.  A 2-cell step draws its stroke through the midpoint cell,
and a station marker there would read as a stop the graph does not have.  Every
2-step midpoint is reserved during growth (rejecting boards that contain one
would select on geometry), and phantom_count checks it.

CANDIDATES.  Six stations (A, B and four decoys) get an identical neutral grey
ring and their names at 29 pt; short_name and CAND_MIN_SEP give the geometry
that makes those names exactly three characters and keeps the candidates >= 4
grid cells apart.
"""
from __future__ import annotations

import collections
import random

import networkx as nx


# 8 octilinear unit directions
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _turn(cdir, k):
    return DIRS[(DIRS.index(cdir) + k) % 8]


GRID = 16              # legibility: on 16x16 the pitch floor is 317/15 = 21.13px = 1.51
                       # patch.  A 17x17 grid gives 19.8px = 1.41 patch, under 1.5.
                       # The grid is a CONSTRAINT here, not a difficulty knob.
CHORD_DELTA = (1, 2, 3, 4)   # chord is this many stations SHORTER than the
                             # existing X-Y route
CUTOFF = 14                  # route enumeration cutoff, L* + CUTOFF edges
N_CANDIDATES = 6             # ringed / big-labelled stations: A, B + 4 decoys
CAND_MIN_SEP = 4             # cells, Chebyshev, between any two candidates.
                             # The big-label convention places a name at
                             # (dot + 11, ...) or (dot - width, ...); with a
                             # 55px name that is 66px, and 3 cells is 63px, so
                             # two candidates 3 cells apart can have the same
                             # name box and the name stops identifying its dot.


# ---------------------------------------------------------------- grower
def placeable(c, occ, res, mine, mymids):
    """May a STATION sit at cell c?

    Never on a stroke midpoint -- `res` holds every cell an earlier 2-cell step
    draws its stroke through, `mymids` the current line's.  That is the phantom
    fix: a station there is painted on top of a stroke and reads as a stop the
    graph has no node for.  A cell already carrying another line's station IS
    allowed (that is the interchange), and the crowding check is skipped for it.
    """
    if not (0 <= c[0] < GRID and 0 <= c[1] < GRID):
        return False
    if c in res or c in mymids or c in mine:
        return False
    if c in occ:
        return True
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            if (c[0] + dx, c[1] + dy) in occ:
                return False
    return True


def grow(rng, occ, res, n, start, d0):
    if start in res or start in occ or not (0 <= start[0] < GRID and 0 <= start[1] < GRID):
        return None
    cells = [start]
    mine = {start}
    mymids = set()
    cur, cdir = start, d0
    guard = 0
    while len(cells) < n and guard < 400:
        guard += 1
        if rng.random() < 0.30 and len(cells) >= 2:
            cdir = _turn(cdir, rng.choice([-1, 1]))
        ok = False
        nxt = mid = None
        for _ in range(8):
            step = 2 if rng.random() < 0.55 else 1
            nxt = (cur[0] + cdir[0] * step, cur[1] + cdir[1] * step)
            mid = (cur[0] + cdir[0], cur[1] + cdir[1]) if step == 2 else None
            # a midpoint may never be a station -- mine or anyone else's
            if mid is not None and (mid in occ or mid in mine):
                pass
            elif placeable(nxt, occ, res, mine, mymids):
                ok = True
                break
            cdir = _turn(cdir, rng.choice([-1, 1, 2, -2]))
        if not ok:
            break
        if mid is not None:
            mymids.add(mid)
        cells.append(nxt)
        mine.add(nxt)
        cur = nxt
    return (cells, mymids) if len(cells) >= 2 else None


SEEDS = [((1, 1), (1, 1)), ((GRID - 2, 1), (-1, 1)), ((1, GRID - 2), (1, -1)),
         ((GRID - 2, GRID - 2), (-1, -1)), ((GRID // 2, 1), (0, 1)),
         ((1, GRID // 2), (1, 0))]


def base_lines(rng, n_lines, lo, hi):
    occ, res, lines = set(), set(), []
    seeds = list(SEEDS)
    rng.shuffle(seeds)
    for li in range(n_lines):
        want = rng.randint(lo, hi)
        best = None
        s0, d0 = seeds[li % len(seeds)]
        for _ in range(60):
            st = (max(1, min(GRID - 2, s0[0] + rng.randint(-2, 2))),
                  max(1, min(GRID - 2, s0[1] + rng.randint(-2, 2))))
            g = grow(rng, occ, res, want, st, d0)
            if g is None or len(g[0]) < lo:
                continue
            shared = sum(1 for c in g[0] if c in occ)
            score = (len(g[0]) if li == 0 else
                     200 + len(g[0]) - 3 * shared if 1 <= shared <= 2 else
                     40 + len(g[0]) if shared == 0 else 8)
            if best is None or score > best[0]:
                best = (score, g)
        if best is None:
            return None
        cells, mids = best[1]
        occ |= set(cells)
        res |= set(mids)
        lines.append(cells)
    return lines, occ, res


def grow_chord(X, Y, occ, res, rng, want, maxexp=15000):
    """BFS for an X->Y polyline with EXACTLY `want` intermediate stations."""
    seen = {(X, 0)}
    q = collections.deque([(X, 0, [X], [])])
    exp = 0
    while q:
        cur, k, path, mids = q.popleft()
        exp += 1
        if exp > maxexp:
            return None
        cand = []
        for d in DIRS:
            for step in (1, 2):
                nxt = (cur[0] + d[0] * step, cur[1] + d[1] * step)
                mid = (cur[0] + d[0], cur[1] + d[1]) if step == 2 else None
                if mid is not None and (mid in occ or mid in path):
                    continue
                if nxt == Y:
                    # Y is reserved for the endpoint.  Letting it into the
                    # intermediate path and then taking this shortcut later
                    # revisits one physical station (and can retrace track).
                    if k == want and Y not in path:
                        return path + [Y], mids + ([mid] if mid else [])
                    continue
                if k >= want or nxt in mids:
                    continue
                if not placeable(nxt, occ, res, set(path), set(mids)):
                    continue
                cand.append((nxt, k + 1, path + [nxt], mids + ([mid] if mid else [])))
        rng.shuffle(cand)
        for c in cand:
            if (c[0], c[1]) in seen:
                continue
            seen.add((c[0], c[1]))
            q.append(c)
    return None


def graph_of(lines):
    G = nx.Graph()
    for li, cells in enumerate(lines):
        for c in cells:
            G.add_node(c)
            G.nodes[c].setdefault("lines", set()).add(li)
        for a, b in zip(cells, cells[1:]):
            if G.has_edge(a, b):
                G[a][b]["lines"].add(li)
            else:
                G.add_edge(a, b, lines={li})
    return G


def phantom_count(lines):
    """Segments whose 2-cell step passes through a cell that carries a station.
    Must be 0 by construction; asserted, not hoped for."""
    st = set()
    for l in lines:
        st |= set(l)
    n = 0
    for l in lines:
        for a, b in zip(l, l[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if max(abs(dx), abs(dy)) == 2 and (a[0] + dx // 2, a[1] + dy // 2) in st:
                n += 1
    return n


# ---------------------------------------------------------------- the scalar
def transfers_on(G, p):
    el = [G[p[i]][p[i + 1]]["lines"] for i in range(len(p) - 1)]
    if not el:
        return 0
    t, cur = 0, set(el[0])
    for e in el[1:]:
        if cur & e:
            cur &= e
        else:
            t += 1
            cur = set(e)
    return t


def axis(G, Gc, A, B, d):
    """kappa and friends for one (A, B).  d = BFS distance in edges."""
    ps = list(nx.all_simple_paths(Gc, A, B, cutoff=d + CUTOFF))
    Ls = [len(p) - 2 for p in ps]
    Ts = [transfers_on(G, p) for p in ps]
    Lstar = d - 1
    Tmin = min(Ts)
    return dict(
        Lstar=Lstar, npaths=len(ps), nlen=len({*Ls}),
        kappa=sum(Ls) / len(Ls) - Lstar,
        kappa_ta=min(L for L, T in zip(Ls, Ts) if T == Tmin) - Lstar,
        Tstar=min(T for L, T in zip(Ls, Ts) if L == Lstar),
        mu=Gc.number_of_edges() - Gc.number_of_nodes() + 1,
    )


def dfs_budget(G, A, B, k):
    """The shortcut, on the label graph: DFS in a fixed neighbour order, keep the
    best of the first k routes it stumbles into."""
    st = [(A, [A])]
    found = []
    while st and len(found) < k:
        v, p = st.pop()
        if v == B:
            found.append(len(p) - 2)
            continue
        for w in sorted(G.neighbors(v), reverse=True):
            if w not in p:
                st.append((w, p + [w]))
    return min(found) if found else None


def cheb(u, v):
    return max(abs(u[0] - v[0]), abs(u[1] - v[1]))


# ---------------------------------------------------------------- names
# pronounceable nonsense names -- kept SHORT (1-2 syllables, e.g. "Quis", "Sit",
# "Nec", "Metu") so labels stay legible on the map.
_ONSET = ["b", "d", "f", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "kr", "tr", "st", "pl", "gl", "sn", "qu"]
_VOWEL = ["a", "e", "i", "o", "u"]
_CODA = ["", "", "", "n", "s", "l", "r", "x", "m", "t"]


def gen_name(rng: random.Random) -> str:
    syl = rng.choice([1, 1, 2])  # mostly short
    s = ""
    for _ in range(syl):
        s += rng.choice(_ONSET) + rng.choice(_VOWEL) + rng.choice(_CODA)
    return s.capitalize()


_ON1 = [o for o in _ONSET if len(o) == 1]
_CO1 = [c for c in _CODA if len(c) == 1]


def short_name(rng, used):
    """Exactly three characters for a candidate.  The six names are the only
    ones drawn at 29 pt (cap height 21px = 1.5 patch), and a 29pt label is
    as tall as the station pitch, so its WIDTH is what decides how much track it
    buries: a 4-char name (75px) can bury a whole 21px inter-station run, which
    removes the answer rather than making it harder; 3-char names are 55px.  455 distinct names, 6 needed per board."""
    for _ in range(500):
        nm = (rng.choice(_ON1) + rng.choice(_VOWEL) + rng.choice(_CO1)).capitalize()
        if nm not in used:
            used.add(nm)
            return nm
    raise RuntimeError("name exhaustion")
