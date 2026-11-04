"""metro_map: one image, several questions; five rungs; nine idioms.

A BLOCK is one board, drawn once, with six ringed candidate stations.  It
yields four records that share that image and differ only in what is asked:

    min_stops       fewest stops from A to B, transfers free
    min_transfers   fewest line changes from X to Y
    line_stations   how many stations a named line serves
    nearest_ring    which other ringed station is fewest stops from X

The rung is a property of the block.  It moves two things and nothing else:
the board recipe (how much map there is to read) and the min_stops pair rule
(whether "minimum" has any work to do, metro_board's kappa):

    tier    lines  chords  stations   min_stops pair
    easy    3-4    0-1     14-30      kappa 0, one route length, L* 4-7, T* 0 or 1
    medium  4-5    1-2     28-42      kappa 0, one route length, L* 7-10, T* >= 1
    hard    5-6    2       38-50      kappa 0, one route length, L* 8-14, T* >= 1
    xhard   5-6    2       38-50      kappa 1.5-3.0, >= 3 route lengths, L* 8-14
    xxhard  5-6    2       38-50      kappa >= 4.5, >= 5 route lengths, L* 8-14, and the
                                      fewest-transfer route is not the fewest-stop route

hard, xhard and xxhard share one board recipe; easy and medium use the smaller
recipes above.

train_id blocks are easy/medium/hard/xhard in the five training idioms;
ood_eval blocks are xxhard in the four held-out idioms.  Style, line count
and wording are scheduled independently of the rung within each role.

Determinism: a block is a pure function of (seed, dataset_role, block).
Nuisances are drawn once per block before the retry loop; only the board
geometry is redrawn on rejection, so a rejection correlated with a nuisance
cannot silently select it.
"""
from __future__ import annotations
from visro.seeding import seed_text

import dataclasses
import hashlib
import heapq
import math
import random
from typing import Any

import networkx as nx

from . import metro_board, metro_render

RNG_NAMESPACE = seed_text(755025569715236233320309650139227321537498001523213201521074831682199346)  # fixed RNG namespace: changing it changes every map
SIZE = 448
TRAIN_TIERS = ("easy", "medium", "hard", "xhard")
OOD_TIER = "xxhard"
DATASET_ROLES = ("train_id", "ood_eval")
FORMS = ("min_stops", "min_transfers", "line_stations", "nearest_ring")
RECORDS_PER_BLOCK = len(FORMS)          # index = block * 4 + slot
MAX_ATTEMPTS = 400
MAX_STATIONS = 50

ID_STYLES = tuple(s for s in metro_render.STYLES if s.role == "id")
OOD_STYLES = tuple(s for s in metro_render.STYLES if s.role == "ood")
STYLE_BY_NAME = {s.name: s for s in metro_render.STYLES}
assert len(ID_STYLES) == 5 and len(OOD_STYLES) == 4

# tier -> (line counts, chord counts, base-line length per line count, min stations)
# hard / xhard / xxhard share one recipe (5-6 lines, two chords, 38+ stations):
# it is the floor, and nothing above medium may be easier than it.  easy and medium are added BELOW it.
_TOP = dict(lines=(5, 6), chords=(2,), base_len={5: (13, 16), 6: (11, 13)}, min_st=38)
# easy must not sit too far below hard (3-4 lines, at most one chord), and
# medium sits in the gap: 4-5 lines, one or two chords, 28-42 stations.
RECIPES = {
    "easy":   dict(lines=(3, 4), chords=(0, 1), base_len={3: (7, 10), 4: (6, 9)}, min_st=14, max_st=30),
    # 4 lines with 2 chords is 2 base lines + 2 short chords and usually lands
    # under 28 stations, so the chord count follows the line count here
    "medium": dict(lines=(4, 5), chords=(1, 2), chords_by_lines={4: (1,), 5: (1, 2)},
                   base_len={4: (9, 12), 5: (8, 10)}, min_st=28, max_st=42),
    "hard":   _TOP,
    "xhard":  _TOP,
    "xxhard": _TOP,
}
LSTAR_BAND = {"easy": (4, 5, 6, 7), "medium": (7, 8, 9, 10), "long": (8, 9, 10, 11, 12, 13, 14)}
# On two-chord boards, pairs needing 4 changes hardly exist and 3 is
# scarce, so the top three tiers share (1, 2, 3); the scheduled target cycles
# through the support and a decoy is steered to it (see _pick_decoys).
TRANSFER_SUPPORT = {"easy": (0, 1), "medium": (1, 2), "hard": (1, 2, 3), "xhard": (1, 2, 3), "xxhard": (1, 2, 3)}
COLOUR_NAMES = {"classic": ("red", "yellow", "teal", "purple", "green", "blue"),
                "official_cn": ("red", "blue", "green", "purple", "orange", "cyan"),
                "pastel": ("blue", "green", "orange", "pink", "tan", "magenta")}

TEMPLATES = {
    "min_stops": (
        'In this subway map, what is the minimum number of stations to pass through '
        'from "{a}" station to "{b}" station? (Transfers are allowed, excluding the '
        'origin and destination stations)',
        'In this subway map, what is the minimum number of stops between "{a}" station '
        'and "{b}" station? (Transfers are allowed, excluding the starting and ending '
        'stations)',
        'This is a subway line map, how many stops are there at minimum from "{a}" '
        'station to "{b}" station? (Transfers are allowed, not counting the starting '
        'and ending stations)',
    ),
    "min_transfers": (
        'In this subway map, what is the minimum number of transfers needed to travel from "{a}" '
        'station to "{b}" station? (A transfer means changing from one line to another. Answer with a single number)',
        'On this subway map, how many times must you change lines, at minimum, to get from "{a}" station '
        'to "{b}" station? (Answer with a single number; 0 if one line connects them)',
    ),
    "line_stations": (
        'In this subway map, how many stations does {line} serve in total? (Count every station on that line, '
        'including transfer stations and both end stations. Answer with a single number)',
        'How many stops are there on {line} in this subway map, counting both ends and every transfer station? '
        '(Answer with a single number)',
    ),
    "nearest_ring": (
        'Six stations on this subway map are marked with a grey ring. Among the other five ringed stations, '
        'which one is the fewest stops away from "{a}" station? (Transfers are allowed. Answer with the station name)',
        'Of the ringed stations other than "{a}", which can be reached from "{a}" station with the fewest '
        'stops in this subway map? (Transfers are allowed. Answer with the station name)',
    ),
}


def in_tier(tier: str, L: int, k: float, nlen: int, T: int, kta: float) -> bool:
    if tier == "easy":
        return k < 1e-9 and nlen == 1 and 4 <= L <= 7 and T <= 1
    if tier == "medium":
        return k < 1e-9 and nlen == 1 and 7 <= L <= 10 and T >= 1
    if tier == "hard":
        return k < 1e-9 and nlen == 1 and 8 <= L <= 14 and T >= 1
    if tier == "xhard":
        return 1.5 <= k <= 3.0 and nlen >= 3 and 8 <= L <= 14 and T >= 1
    if tier == "xxhard":
        return k >= 4.5 and nlen >= 5 and 8 <= L <= 14 and T >= 1 and kta > 0
    raise ValueError(tier)


@dataclasses.dataclass(frozen=True)
class BlockSpec:
    block: int
    tier: str
    style: str
    n_lines: int
    n_chords: int
    lstar_target: int
    want_ta: bool
    dataset_role: str


def block_spec(block: int, dataset_role: str, seed: int) -> BlockSpec:
    """Tier and style are cycled (4 and 5 are coprime, so every tier meets
    every style once per 20 blocks); line and chord counts are drawn once per
    block from the tier's recipe."""
    if block < 0:
        raise ValueError("block must be non-negative")
    nrng = random.Random(f"{RNG_NAMESPACE}|nuisance|{seed}|{dataset_role}|{block}")
    if dataset_role == "train_id":
        tier = TRAIN_TIERS[block % 4]
        style = ID_STYLES[block % 5].name
    elif dataset_role == "ood_eval":
        tier = OOD_TIER
        style = OOD_STYLES[block % 4].name
    else:
        raise ValueError(f"unknown dataset role {dataset_role!r}")
    rc = RECIPES[tier]
    n_lines = nrng.choice(rc["lines"])
    n_chords = nrng.choice(rc.get("chords_by_lines", {}).get(n_lines, rc["chords"]))
    if tier == "easy":
        lstar = LSTAR_BAND["easy"][block % 3]
    elif tier == "medium":
        lstar = LSTAR_BAND["medium"][block % 3]
    else:
        lstar = LSTAR_BAND["long"][block % 7]
    # want_ta cycles over the tier's OWN occurrences: train tiers repeat every
    # four blocks, so `block % 2` would give xhard (blocks 3, 7, 11, ...) a
    # constant want_ta.
    occurrence = block // len(TRAIN_TIERS) if dataset_role == "train_id" else block
    return BlockSpec(block, tier, style, n_lines, n_chords, lstar, occurrence % 2 == 0, dataset_role)


# ------------------------------------------------------------------ boards
def build_board(rng: random.Random, spec: BlockSpec):
    rc = RECIPES[spec.tier]
    n_base = spec.n_lines - spec.n_chords
    lo, hi = rc["base_len"][spec.n_lines]
    bl = metro_board.base_lines(rng, n_base, lo, hi)
    if bl is None:
        return None, "base"
    lines, occ, res = bl
    G = metro_board.graph_of(lines)
    for _ in range(spec.n_chords):
        comp = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(comp)
        nodes = sorted(comp)
        ok = False
        for _t in range(90):
            X, Y = rng.sample(nodes, 2)
            try:
                dt = nx.shortest_path_length(Gc, X, Y)
            except nx.NetworkXNoPath:
                continue
            if dt < 4:
                continue
            want = dt - 1 - rng.choice(metro_board.CHORD_DELTA)
            if want < 1:
                continue
            ch = metro_board.grow_chord(X, Y, occ, res, rng, want)
            if ch is None:
                continue
            cells, mids = ch
            lines.append(cells)
            occ |= set(cells)
            res |= set(mids)
            G = metro_board.graph_of(lines)
            ok = True
            break
        if not ok:
            return None, "chord"
    if metro_board.phantom_count(lines):
        return None, "PHANTOM_LEAKED"
    if len(occ) < rc["min_st"]:
        return None, "small"
    if len(occ) > rc.get("max_st", MAX_STATIONS):
        return None, "big"
    return (lines, occ, G), None


def min_stops_pool(G, Gc, spec: BlockSpec, rng: random.Random, cap: int = 400) -> list:
    """(A, B, axis) pairs satisfying the tier's min_stops rule.  Exact target
    L* for the long-route tiers; nearest in-band L* for easy/medium."""
    band = LSTAR_BAND["easy" if spec.tier == "easy" else "medium" if spec.tier == "medium" else "long"]
    dists = {L + 1 for L in band} if spec.tier in ("easy", "medium") else {spec.lstar_target + 1}
    cand = []
    nodes = sorted(Gc.nodes)
    for A in nodes:
        for B, d in nx.single_source_shortest_path_length(Gc, A, cutoff=max(dists)).items():
            if d in dists and A < B and metro_board.cheb(A, B) >= metro_board.CAND_MIN_SEP:
                cand.append((A, B, d))
    rng.shuffle(cand)
    ok = []
    for A, B, d in cand[:cap]:
        a = metro_board.axis(G, Gc, A, B, d)
        if in_tier(spec.tier, a["Lstar"], a["kappa"], a["nlen"], a["Tstar"], a["kappa_ta"]):
            ok.append((A, B, a))
    if spec.tier in ("easy", "medium") and ok:
        best = min(abs(a["Lstar"] - spec.lstar_target) for _, _, a in ok)
        ok = [x for x in ok if abs(x[2]["Lstar"] - spec.lstar_target) == best]
    if spec.tier == "easy" and ok:
        # alternate same-line (T* = 0) and one-transfer routes where the board offers both
        want_T = 0 if spec.want_ta else 1
        pref = [x for x in ok if x[2]["Tstar"] == want_T]
        ok = pref or ok
    return ok


# ------------------------------------------------------------------ the other questions
def min_transfers(G, A, B) -> int:
    """Fewest line changes over ALL routes (stops are free).  Dijkstra on
    (station, line) states; boarding at A on any of its lines is free."""
    best = {}
    pq = [(0, A, l) for l in G.nodes[A]["lines"]]
    heapq.heapify(pq)
    while pq:
        t, v, l = heapq.heappop(pq)
        if (v, l) in best:
            continue
        best[(v, l)] = t
        if v == B:
            return t
        for w in G.neighbors(v):
            for l2 in G[v][w]["lines"]:
                heapq.heappush(pq, (t + (0 if l2 == l else 1), w, l2))
    raise nx.NetworkXNoPath(f"{A} -> {B}")


def _name_everything(G, cands, prng: random.Random) -> dict:
    names, used = {}, set()
    for c in cands:
        names[c] = metro_board.short_name(prng, used)
    for c in G.nodes:
        if c in names:
            continue
        while True:
            nm = metro_board.gen_name(prng)
            if nm not in used:
                used.add(nm)
                names[c] = nm
                break
    return names


def _pick_decoys(Gc, A, B, path, prng: random.Random, near_route: bool,
                 G=None, transfer_target: int | None = None) -> list:
    """Four decoys >= 4 cells apart (falling back to 3, 2, 1),
    off the A-B route where possible.  When a transfer target is given, the
    first decoy is one whose fewest-transfer count from A or B equals it, so
    the min_transfers question can hit its scheduled answer: left to chance,
    the six candidates rarely contained a 3-transfer pair and that answer's
    best-constant baseline sat at 0.5-0.8."""
    pool = [c for c in Gc.nodes if c not in (A, B)]
    if near_route:
        # grown by 5 cells, not 3: with 3 the crop boards fell back to 3-cell
        # candidate spacing and two 29pt names then overlapped
        xs = [c[0] for c in path]
        ys = [c[1] for c in path]
        pool = [c for c in pool if min(xs) - 5 <= c[0] <= max(xs) + 5 and min(ys) - 5 <= c[1] <= max(ys) + 5]
    on_path = set(path)
    pool = sorted(pool, key=lambda c: (c in on_path, c))      # off-route first
    prng.shuffle(pool)
    pool.sort(key=lambda c: c in on_path)
    cands = [A, B]
    if transfer_target is not None and G is not None:
        hits = [c for c in pool
                if nx.shortest_path_length(Gc, A, c) >= 2 and nx.shortest_path_length(Gc, B, c) >= 2
                and (min_transfers(G, A, c) == transfer_target or min_transfers(G, B, c) == transfer_target)]
        for sep in (metro_board.CAND_MIN_SEP, 3):
            steer = [c for c in hits if all(metro_board.cheb(c, q) >= sep for q in cands)]
            if steer:
                cands.append(steer[0])
                break
    for sep in (metro_board.CAND_MIN_SEP, 3, 2, 1):
        for c in pool:
            if len(cands) >= metro_board.N_CANDIDATES:
                break
            if c not in cands and all(metro_board.cheb(c, q) >= sep for q in cands):
                cands.append(c)
        if len(cands) >= metro_board.N_CANDIDATES:
            break
    prng.shuffle(cands)
    return cands


def _nearest_askable(Gc, cands) -> bool:
    """True if at least one candidate has a unique nearest other candidate."""
    for X in cands:
        d = nx.single_source_shortest_path_length(Gc, X)
        ds = sorted(d[c] for c in cands if c != X)
        if len(ds) >= 2 and ds[0] != ds[1]:
            return True
    return False


# ------------------------------------------------------------------ blocks
@dataclasses.dataclass
class Block:
    spec: BlockSpec
    lines: list
    G: Any
    Gc: Any
    A: tuple
    B: tuple
    axis: dict
    ta_hit: bool | None
    cands: list
    names: dict
    path: list
    image: Any
    diag: dict
    attempts: int
    rejections: dict


def sample_block(seed: int, block: int, dataset_role: str, size: int = SIZE) -> Block:
    spec = block_spec(block, dataset_role, seed)
    rej: dict[str, int] = {}
    for att in range(MAX_ATTEMPTS):
        rng = random.Random(f"{RNG_NAMESPACE}|board|{seed}|{dataset_role}|{block}|{att}")
        b, why = build_board(rng, spec)
        if b is None:
            rej[why] = rej.get(why, 0) + 1
            continue
        lines, occ, G = b
        comp = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(comp)
        pool = min_stops_pool(G, Gc, spec, rng)
        if not pool:
            rej["no_pair"] = rej.get("no_pair", 0) + 1
            continue
        ta_hit = None
        if spec.tier == "xhard":
            # want_ta is scheduled 50/50 and is a REQUIREMENT: a board that
            # cannot supply a matching pair is redrawn.
            want = [x for x in pool if (x[2]["kappa_ta"] > 0) == spec.want_ta]
            if not want:
                rej["no_ta_match"] = rej.get("no_ta_match", 0) + 1
                continue
            ta_hit = True
            pool = want
        prng = random.Random(f"{RNG_NAMESPACE}|pair|{seed}|{dataset_role}|{block}")
        A, B, a = pool[prng.randrange(len(pool))]
        path = nx.shortest_path(Gc, A, B)
        style = STYLE_BY_NAME[spec.style]
        support = TRANSFER_SUPPORT[spec.tier]
        occurrence = block // len(TRAIN_TIERS) if dataset_role == "train_id" else block
        # Decoys are re-drawn until some ringed station has a UNIQUE nearest
        # other ring, so nearest_ring can always be asked.
        for _draw in range(20):
            cands = _pick_decoys(Gc, A, B, path, prng, near_route=(style.framing == "crop"),
                                 G=G, transfer_target=support[occurrence % len(support)])
            if _nearest_askable(Gc, cands):
                break
        else:
            rej["no_unique_nearest"] = rej.get("no_unique_nearest", 0) + 1
            continue
        names = _name_everything(G, cands, prng)
        cell_lines = {c: set(G.nodes[c]["lines"]) for c in G.nodes}
        board = metro_render.Board(lines=[list(cs) for cs in lines], cell_lines=cell_lines, names=names,
                         cands=list(cands), A=A, B=B, path=list(path), answer=a["Lstar"])
        img, names_used, diag = metro_render.render_styled(
            board, style, size, random.Random(f"{RNG_NAMESPACE}|render|{seed}|{dataset_role}|{block}"))
        blk = Block(spec, [list(cs) for cs in lines], G, Gc, A, B, a, ta_hit,
                    [tuple(c) for c in diag["candidates"]], names_used, list(path), img, diag,
                    att + 1, rej)
        if style.framing == "crop":
            try:
                # The renderer's crop gate only covers min_stops.
                # A board must support all four forms using what is visible.
                questions_for(blk, seed)
            except UnaskableBoard:
                rej["no_visible_question"] = rej.get("no_visible_question", 0) + 1
                continue
        return blk
    raise RuntimeError(f"no board for block {block} ({dataset_role}) after {MAX_ATTEMPTS} attempts: {rej}")


def _line_reference(nuis: dict, palette_name: str, k: int) -> tuple[str, str]:
    """How the question names line k: by its legend label as drawn, or by the
    colour it was actually given when the image has no legend."""
    if nuis["legend"] != "none":
        return f'the line labelled "{nuis["line_name"].format(n=k + 1)}" in the legend', "legend_label"
    colour = COLOUR_NAMES[palette_name][nuis["palette_order"][k]]
    return f"the {colour} line", "colour"


class UnaskableBoard(ValueError):
    """The rendered board cannot support all of its question forms."""


def visible_graph(G, station_px: dict, size: int = SIZE):
    """Graph induced by fully visible station centres, matching the renderer.

    Crop framing rejects windows that cut through a marker.  Edges retain
    their line memberships; in particular, disconnected visible portions of
    one line do not become connected just because their colours match.
    """
    visible = {tuple(map(int, c.split(","))) for c, (x, y) in station_px.items()
               if 3 <= x <= size - 3 and 3 <= y <= size - 3}
    return G.subgraph(visible)


def questions_for(blk: Block, seed: int) -> list[dict]:
    """The records a block yields: (form, question, answer, cot, form meta)."""
    spec, G, Gc = blk.spec, blk.G, blk.Gc
    A, B, names, cands = blk.A, blk.B, blk.names, blk.cands
    style = STYLE_BY_NAME[spec.style]
    if len(cands) != metro_board.N_CANDIDATES or len(set(cands)) != len(cands) or any(c not in Gc for c in cands):
        # Crop rendering may reselect decoys from a disconnected part of the
        # board.  Reject that draw rather than asking an undefined distance.
        raise UnaskableBoard("the six ringed stations must share the question's connected component")
    out = []
    trng = random.Random(f"{RNG_NAMESPACE}|template|{seed}|{spec.dataset_role}|{spec.block}")
    dist = {c: nx.single_source_shortest_path_length(Gc, c) for c in cands}
    visible = visible_graph(G, blk.diag["station_px"], blk.image.width)
    if any(c not in visible for c in cands):
        raise UnaskableBoard("a ringed station is outside the visible graph")
    cropped = len(visible) != len(G)

    # 1. min_stops -- the tier's pair
    L = blk.axis["Lstar"]
    if cropped:
        try:
            if nx.shortest_path_length(visible, A, B) - 1 != L:
                raise UnaskableBoard("min_stops changes in the visible graph")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise UnaskableBoard("min_stops is disconnected in the visible graph") from exc
    route_cot = (f'The shortest route visible in the image passes through {L + 2} stations in total, '
                 if style.framing == "crop" else
                 f'There are {blk.axis["npaths"]} route(s) with {blk.axis["nlen"]} distinct stop count(s); '
                 f'the shortest passes through {len(blk.path)} stations in total, ')
    t = trng.randrange(len(TEMPLATES["min_stops"]))
    out.append(dict(form="min_stops", template=t,
                    question=TEMPLATES["min_stops"][t].format(a=names[A], b=names[B]),
                    answer=str(L), targets=dict(A=A, B=B),
                    cot=(f'"{names[A]}" and "{names[B]}" are two of the six ringed stations. '
                         f'{route_cot}so excluding origin and '
                         f'destination the answer is {L}.')))

    # 2. min_transfers -- a pair among the six with a transfer count in the tier's support
    support = TRANSFER_SUPPORT[spec.tier]
    pairs = []
    for i, X in enumerate(cands):
        for Y in cands[i + 1:]:
            if dist[X].get(Y, 0) >= 2:
                transfers = min_transfers(G, X, Y)
                if cropped:
                    try:
                        if min_transfers(visible, X, Y) != transfers:
                            continue
                    except nx.NetworkXNoPath:
                        continue
                pairs.append((X, Y, transfers))
    # cycle the target over the tier's OWN occurrences: train tiers repeat
    # every 4 blocks, so block % len(support) would pin easy to 0 forever
    occurrence = spec.block // len(TRAIN_TIERS) if spec.dataset_role == "train_id" else spec.block
    target = support[occurrence % len(support)]
    pick = [p for p in pairs if p[2] == target] or [p for p in pairs if p[2] in support]
    if not pick:
        pick = [p for p in pairs if p[:2] in ((A, B), (B, A))] or pairs
    if not pick:
        raise UnaskableBoard("no min_transfers pair agrees with the visible graph")
    X, Y, T = pick[random.Random(f"{RNG_NAMESPACE}|tpair|{seed}|{spec.dataset_role}|{spec.block}").randrange(len(pick))]
    t = trng.randrange(len(TEMPLATES["min_transfers"]))
    out.append(dict(form="min_transfers", template=t,
                    question=TEMPLATES["min_transfers"][t].format(a=names[X], b=names[Y]),
                    answer=str(T), targets=dict(A=X, B=Y, target_in_support=T in support),
                    cot=(f'From "{names[X]}" to "{names[Y]}" every route needs at least {T} line change(s); '
                         f'a route with {T} transfer(s) exists, so the answer is {T}.')))

    # 3. line_stations -- one line, named by legend label or colour
    occurrence = spec.block // len(TRAIN_TIERS) if spec.dataset_role == "train_id" else spec.block
    k = occurrence % len(blk.lines)
    visible_lines = [i for i, cs in enumerate(blk.lines) if all(c in visible for c in cs)]
    if k not in visible_lines:
        if not visible_lines:
            raise UnaskableBoard("no entire line is visible for line_stations")
        k = visible_lines[occurrence % len(visible_lines)]
    ref, ref_mode = _line_reference(blk.diag["nuisance"], style.palette, k)
    n_on_line = len(set(blk.lines[k]))
    t = trng.randrange(len(TEMPLATES["line_stations"]))
    out.append(dict(form="line_stations", template=t,
                    question=TEMPLATES["line_stations"][t].format(line=ref),
                    answer=str(n_on_line), targets=dict(line_index=k, line_reference=ref_mode),
                    cot=(f'Counting each distinct station on {ref} once gives {n_on_line} stations, '
                         f'so the answer is {n_on_line}.')))

    # 4. nearest_ring -- an anchor whose nearest other ring is unique; prefer one
    #    where the pixel-nearest ring is NOT the stop-nearest (so distance on the
    #    page does not answer the question)
    px = {c: tuple(blk.diag["station_px"][f"{c[0]},{c[1]}"]) for c in cands}
    options = []
    for X in cands:
        others = [c for c in cands if c != X]
        ds = sorted((dist[X][c], c) for c in others)
        if ds[0][0] == ds[1][0]:
            continue
        stop_nearest = ds[0][1]
        if cropped:
            vd = nx.single_source_shortest_path_length(visible, X)
            vds = sorted((vd.get(c, math.inf), c) for c in others)
            if vds[0][0] == vds[1][0] or vds[0] != ds[0]:
                continue
        pix_nearest = min(others, key=lambda c: math.dist(px[X], px[c]))
        options.append((pix_nearest != stop_nearest, X, stop_nearest, ds[0][0]))
    if options:
        options.sort(key=lambda o: (not o[0], names[o[1]]))
        adversarial, X, Z, dz = options[0]
        t = trng.randrange(len(TEMPLATES["nearest_ring"]))
        out.append(dict(form="nearest_ring", template=t,
                        question=TEMPLATES["nearest_ring"][t].format(a=names[X]),
                        answer=names[Z], targets=dict(A=X, nearest=Z, stops=dz - 1,
                                                      pixel_nearest_differs=adversarial),
                        cot=(f'Counting stops from "{names[X]}" to each other ringed station, "{names[Z]}" is '
                             f'the closest at {dz - 1} intermediate stop(s), so the answer is {names[Z]}.')))
    else:
        raise UnaskableBoard("no unique nearest_ring answer agrees with the visible graph")

    return out


def block_records(seed: int, block: int, dataset_role: str, size: int = SIZE, split: str | None = None,
                  questions_per_image: int = 4):
    """The records of a block plus its one image.  `split` names the corpus
    the block belongs to (train / eval / val / ood) and goes into ids and image
    names, so two corpora drawn from the same role with different seeds cannot
    collide when they are loaded together.

    questions_per_image=4 emits every form on the image; =1 keeps one form per
    board, rotating over the tier's (or, for OOD, the style's) successive
    boards so tiers, styles and forms stay balanced, and ids are the block
    index.  All four are still built and validated; three are discarded."""
    blk = sample_block(seed, block, dataset_role, size)
    spec = blk.spec
    style = STYLE_BY_NAME[spec.style]
    role_tag = split or ("train_id" if dataset_role == "train_id" else "eval_ood")
    image_name = f"images/{role_tag}_b{block:06d}_metro.png"
    image_sha = hashlib.sha256(blk.image.convert("RGB").tobytes()).hexdigest()
    cell_lines = {c: set(blk.G.nodes[c]["lines"]) for c in blk.G.nodes}
    rings = sum(1 for ls in cell_lines.values() if len(ls) >= 2)
    a = blk.axis
    base_meta: dict[str, Any] = {
        "schema_version": RNG_NAMESPACE, "generation_seed": seed, "dataset_role": dataset_role,
        "is_ood": dataset_role == "ood_eval", "split": role_tag,
        "allowed_tiers": list(TRAIN_TIERS if dataset_role == "train_id" else (OOD_TIER,)),
        "block": block, "image_size": size, "tier": spec.tier,
        "style": style.name, "style_role": style.role,
        "appearance": {f.name: (list(getattr(style, f.name)) if isinstance(getattr(style, f.name), tuple)
                                else getattr(style, f.name))
                       for f in dataclasses.fields(style) if f.name not in ("name", "note", "source", "role")},
        "appearance_sampled": blk.diag["nuisance"],
        "recipe": {"n_lines": spec.n_lines, "n_chords": spec.n_chords,
                   "base_len": list(RECIPES[spec.tier]["base_len"][spec.n_lines])},
        "n_lines": len(blk.lines), "n_stations": blk.G.number_of_nodes(), "n_interchanges": rings,
        "min_stops_pair": {"A_cell": list(blk.A), "B_cell": list(blk.B), "A_name": blk.names[blk.A],
                           "B_name": blk.names[blk.B], "kappa": round(a["kappa"], 6), "kappa_ta": a["kappa_ta"],
                           "n_distinct_route_lengths": a["nlen"], "n_routes": a["npaths"], "Tstar": a["Tstar"],
                           "Lstar": a["Lstar"], "Lstar_target": spec.lstar_target,
                           "want_ta": spec.want_ta if spec.tier in ("xhard", "xxhard") else None,
                           "want_ta_hit": blk.ta_hit,
                           "shortest_path_cells": [list(c) for c in blk.path],
                           "dfs_b1": metro_board.dfs_budget(blk.Gc, blk.A, blk.B, 1),
                           "dfs_b3": metro_board.dfs_budget(blk.Gc, blk.A, blk.B, 3)},
        "board_attempts": blk.attempts, "rejections": blk.rejections,
        "candidates": [list(c) for c in blk.cands],
        "cand_min_sep": min(metro_board.cheb(u, v) for i, u in enumerate(blk.cands) for v in blk.cands[i + 1:]),
        "decoys_near_route": style.framing == "crop", "decoys_repicked_by_renderer": blk.diag["decoys_repicked"],
        "lines_cells": [[list(c) for c in cs] for cs in blk.lines],
        "line_memberships": {f"{c[0]},{c[1]}": sorted(ls) for c, ls in cell_lines.items()},
        "station_names": {f"{c[0]},{c[1]}": blk.names[c] for c in cell_lines},
        "station_px": blk.diag["station_px"],
        "phantom_segments": metro_board.phantom_count(blk.lines),
        "render": {k: blk.diag[k] for k in ("pitch_px", "pitch_patches", "labels_dropped",
                                            "candidate_labels_on_track", "candidate_name_collisions", "erased_markers",
                                            "n_visible_stations", "palette_min_pair_de", "palette_min_bg_de")},
        "crop_gate_ok": blk.diag.get("crop_gate_ok"),
        "image_pixel_sha256": image_sha,
    }
    questions = questions_for(blk, seed)
    if questions_per_image == 1:
        # tiers cycle block % 4 in train, styles block % 4 in ood: the form
        # advances once per occurrence of the same tier / style
        occurrence = block // len(TRAIN_TIERS) if dataset_role == "train_id" else block // len(OOD_STYLES)
        want = FORMS[occurrence % len(FORMS)]
        questions = [q for q in questions if q["form"] == want]
        if not questions:
            raise RuntimeError(f"block {block} could not ask {want}")
    elif questions_per_image != len(FORMS):
        raise ValueError("questions_per_image must be 1 or 4")
    records = []
    for slot, q in enumerate(questions):
        index = block if questions_per_image == 1 else block * RECORDS_PER_BLOCK + FORMS.index(q["form"])
        meta = dict(base_meta)
        meta.update({"global_index": index, "slot": FORMS.index(q["form"]), "question_form": q["form"],
                     "questions_per_image": questions_per_image,
                     "template_id": f"{q['form']}_t{q['template']}",
                     "question_targets": {k: (list(v) if isinstance(v, tuple) else v) for k, v in q["targets"].items()}})
        records.append({
            "id": f"metro_map_{role_tag}_{index:08d}", "source_id": f"metro_map_{role_tag}_{index:08d}",
            "task": "metro_map", "task_type": "visual_tracking", "subtype": "metro_map", "level": "mid",
            "difficulty": spec.tier, "tier": spec.tier, "question_form": q["form"], "style": style.name,
            "question": q["question"], "ans_type": "blank", "options": [], "answer": q["answer"], "cot": q["cot"],
            "image": image_name, "meta": meta,
        })
    for r in records:
        validate_record(r, blk.image)
    return records, blk.image


# ------------------------------------------------------------------ validation
def _graph_from_meta(m):
    G = nx.Graph()
    for li, cells in enumerate(m["lines_cells"]):
        cs = [tuple(c) for c in cells]
        for c in cs:
            G.add_node(c)
            G.nodes[c].setdefault("lines", set()).add(li)
        for u, v in zip(cs, cs[1:]):
            if G.has_edge(u, v):
                G[u][v]["lines"].add(li)
            else:
                G.add_edge(u, v, lines={li})
    return G


def validate_record(record: dict, image=None) -> None:
    """Recompute the answer of every form from the stored line cells, and
    re-check the tier rule and the rendering contracts from stored numbers."""
    m = record["meta"]
    role = m["dataset_role"]
    if role not in DATASET_ROLES:
        raise ValueError(f"unknown dataset_role {role!r}")
    if role == "train_id" and record["tier"] not in TRAIN_TIERS:
        raise ValueError("train_id record carries an OOD tier")
    if role == "ood_eval" and record["tier"] != OOD_TIER:
        raise ValueError("ood_eval record is not xxhard")
    style = STYLE_BY_NAME[record["style"]]
    if (style.role == "id") != (role == "train_id"):
        raise ValueError(f"style {style.name} is {style.role} but the record is {role}")
    G = _graph_from_meta(m)
    comp = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(comp)
    names = {tuple(int(x) for x in k.split(",")): v for k, v in m["station_names"].items()}
    tg = m["question_targets"]
    form = record["question_form"]
    p = m["min_stops_pair"]
    A, B = tuple(p["A_cell"]), tuple(p["B_cell"])
    if not in_tier(record["tier"], p["Lstar"], p["kappa"], p["n_distinct_route_lengths"], p["Tstar"], p["kappa_ta"]):
        raise ValueError(f"stored numbers do not satisfy the {record['tier']} rule")
    if nx.shortest_path_length(G, A, B) - 1 != p["Lstar"]:
        raise ValueError("min_stops pair disagrees with the stored line cells")
    if form == "min_stops":
        expected = str(p["Lstar"])
    elif form == "min_transfers":
        expected = str(min_transfers(G, tuple(tg["A"]), tuple(tg["B"])))
    elif form == "line_stations":
        # Count physical graph nodes independently of the producer's line
        # sequence: earlier chords may list an endpoint twice.
        expected = str(sum(tg["line_index"] in ls for _, ls in G.nodes(data="lines")))
    elif form == "nearest_ring":
        X = tuple(tg["A"])
        d = nx.single_source_shortest_path_length(Gc, X)
        others = [tuple(c) for c in m["candidates"] if tuple(c) != X]
        ds = sorted((d[c], c) for c in others)
        if ds[0][0] == ds[1][0]:
            raise ValueError("nearest_ring answer is not unique")
        expected = names[ds[0][1]]
        if tuple(tg["nearest"]) != ds[0][1] or tg["stops"] != ds[0][0] - 1:
            raise ValueError("nearest_ring target or stop count disagrees with the stored line cells")
    else:
        raise ValueError(f"unknown form {form!r}")
    if expected != record["answer"]:
        raise ValueError(f"{form}: answer {record['answer']!r} != recomputed {expected!r}")
    visible = visible_graph(G, m["station_px"], m["image_size"])
    if len(visible) != len(G):
        try:
            if form == "min_stops":
                visible_answer = str(nx.shortest_path_length(visible, tuple(tg["A"]), tuple(tg["B"])) - 1)
            elif form == "min_transfers":
                visible_answer = str(min_transfers(visible, tuple(tg["A"]), tuple(tg["B"])))
            elif form == "line_stations":
                visible_answer = str(sum(tg["line_index"] in ls for _, ls in visible.nodes(data="lines")))
            else:
                d = nx.single_source_shortest_path_length(visible, tuple(tg["A"]))
                ds = sorted((d.get(tuple(c), math.inf), tuple(c)) for c in m["candidates"]
                            if tuple(c) != tuple(tg["A"]))
                if ds[0][0] == ds[1][0]:
                    raise ValueError("nearest_ring answer is not unique in the visible graph")
                visible_answer = names[ds[0][1]]
                if ds[0][0] - 1 != tg["stops"]:
                    raise ValueError("nearest_ring visible stop count disagrees with the stored target")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise ValueError(f"{form}: question is disconnected in the visible graph") from exc
        if visible_answer != expected:
            raise ValueError(f"{form}: visible answer {visible_answer!r} != full-board answer {expected!r}")
    for key in ("A", "B"):
        if key in tg and f'"{names[tuple(tg[key])]}"' not in record["question"]:
            raise ValueError(f"question does not name {key}")
    if m["phantom_segments"]:
        raise ValueError("phantom segments")
    if m["render"]["erased_markers"]:
        raise ValueError("a marker was painted out")
    if m["render"]["pitch_px"] < 21.0:
        raise ValueError("station pitch below the 21px floor")
    if style.framing == "crop" and not m["crop_gate_ok"]:
        raise ValueError("crop window changes the answer")
    cands = [tuple(c) for c in m["candidates"]]
    if len(cands) != metro_board.N_CANDIDATES or A not in cands or B not in cands:
        raise ValueError("candidate set must be six ringed stations including A and B")
    if image is not None and hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != m["image_pixel_sha256"]:
        raise ValueError("image hash disagrees with rendered pixels")
