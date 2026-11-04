"""maze board law, Lambda measure, marker rules and renderer, shared by the
maze generators (maze_core, detour_core, corridor_questions).

AXIS.  Lambda(E) = sum over cells of E of max(0, deg_open - 2), where E is the
minimum certificate region: the smallest cell set any correct algorithm must
expand to PROVE a sample's answer.  Lambda counts binary branch decisions inside
the region the answer depends on.  A degree-2 corridor cell contributes 0; a
4-way junction contributes 2.  This module measures it (excess, lam); the case
builders that choose E live in the modules that import it.

1. ONE BOARD LAW.  One grid, one carver law and one passage law for every
   tier, so the image size carries no information about the tier.

2. THE PASSAGE COUNT IS EXACT.  P = N-1+K with K drawn from one law shared by
   every tier, and the board is built to hit P EXACTLY: carve a spanning tree
   (P = N-1), cut k-1 tree edges to make the k components the task needs
   (P = N-k), then open exactly P-(N-k) intra-component walls.  Total ink is
   therefore a pure function of K and carries zero bits about the tier, by
   construction rather than by tuning.

3. THE BRAID IS SPATIALLY NON-UNIFORM, from a law shared by all tiers.  Extra
   passages are opened with a weight field of 2-4 random hotspots, so every
   board has dense and sparse regions.  That lets the tier ride on QUESTION
   SELECTION (which region the certificate covers) at a fixed board law instead
   of on K, i.e. on the ink.

4. ONE MARKER RULE.  Every marker passes marker_ok (MIN_DEG open passages and
   MIN_DEPTH corridor steps of escape room) and every marker pair is separated
   by SEP of the image diagonal (sep_ok), the same at every tier and for every
   task.

The lattices, carvers and drawing come from lattice.py, the question sentences
and record builders from questions.py.
"""

from __future__ import annotations

import math
import statistics as st

from . import lattice, questions

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
# ONE grid for every tier and every task.  Corridor = 21.9 px at 448 with
# thickness="thick" (the worst of the three the generator samples), marker
# 22.3 px -- both >= 1.5 SigLIP patches.
GRID = (17, 17, 44)                 # w, h, cell   -> N = 289, image 818x818
LATTICE = "rect"

CARVERS = ("dfs", "wilson", "prim", "kruskal")

SEP = (0.12, 0.60)                  # marker separation / image diagonal, SHARED
MIN_DEG = 2                         # every marker, every task, every tier
MIN_DEPTH = 6

TIE_EPS = 0.05                      # |eA-eB|/max below this = straight-line tie


# --------------------------------------------------------------------------- #
# Graph helpers and the marker rule
# --------------------------------------------------------------------------- #
def odeg(lat, walls, c):
    return lattice.open_degree(c, lat.neighbors, walls)


def excess(lat, walls):
    return {c: max(0, odeg(lat, walls, c) - 2) for c in lat.cells}


def lam(ex, cells):
    return sum(ex[c] for c in cells)


def readable_marker_cell(lat, cell):
    """Avoid marker locations that are visually bad even if graph-valid.

    Polar cells near the centre are tiny wedges; a marker there can cover most
    of the cell and look like it is trapped in walls. Keep polar TF markers in
    the outer half of rings.
    """
    if lat.name == "polar":
        return cell[1] >= max(2, math.ceil(lat.height * 0.50))
    return True


def marker_ok(lat, walls, c):
    """The marker rule: a readable cell with real local escape room (MIN_DEG open
    passages, MIN_DEPTH corridor steps), not just long global depth."""
    if not readable_marker_cell(lat, c):
        return False
    if odeg(lat, walls, c) < MIN_DEG:
        return False
    dist = lattice.bfs_dist(c, lat.neighbors, walls)
    return bool(dist) and max(dist.values()) >= MIN_DEPTH


# --------------------------------------------------------------------------- #
# Board construction:  P is EXACT
# --------------------------------------------------------------------------- #
def subtree_edges(lat, walls, comp):
    """(child, parent, subtree_size) for every open edge inside `comp`.
    `comp` must still be a tree -- all cutting happens before any braiding."""
    comp = set(comp)
    root = min(comp)
    parent, order = {root: None}, [root]
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in lat.neighbors(cur):
            if nb in comp and nb not in parent and lattice.wall_key(cur, nb) not in walls:
                parent[nb] = cur
                order.append(nb)
                stack.append(nb)
    size = dict.fromkeys(order, 1)
    for c in reversed(order[1:]):
        size[parent[c]] += size[c]
    return [(c, parent[c], size[c]) for c in order[1:]]


def partition_tree(lat, tree_walls, rng, fracs):
    """Cut len(fracs)-1 tree edges so the components approximate `fracs` * N.

    Cutting TREE edges (k-1 of them) instead of sealing a region's boundary is
    what keeps the certificate's branch count proportional to its size: sealing
    a boundary also strips the branches that leave through it (a 45% region of
    a board with a global Lambda of 84 then measures only Lambda = 14)."""
    N = len(lat.cells)
    walls = set(tree_walls)
    comps = [set(lat.cells)]
    want = sorted(int(round(f * N)) for f in fracs)
    for t in want[:-1]:
        big = max(comps, key=len)
        cand = subtree_edges(lat, walls, big)
        cand = [e for e in cand if 2 <= e[2] <= len(big) - 2]
        if not cand:
            return None, None
        c, p, s = min(cand, key=lambda e: (abs(e[2] - t), rng.random()))
        walls.add(lattice.wall_key(c, p))
        comps.remove(big)
        piece = lattice.reachable(c, lat.neighbors, walls)
        comps.append(piece)
        comps.append(big - piece)
    return walls, comps


def braid_field(lat, rng, focus=None):
    """2-4 random hotspots.  Shared law: identical at every tier.

    `focus` (a component, chosen UNIFORMLY at random by the caller and never by
    the tier) puts most of the hotspots inside that component.  Without it a
    component's branch count is essentially its cell count times the global
    density, so a high band can only be reached by a LARGE component -- and the
    cue "answer = the start in the biggest component" then rises with the tier.
    Without the focus that cue scores 0.358 / 0.417 / 0.533 on the three tiers
    of the which-reaches question, against a best constant of 0.250.  With the
    focus the size and the branch count come apart.  The price is a new cue,
    "the start in the DENSEST component"; unlike component size it cannot be
    read without counting junctions, which is the work the task asks for."""
    W, Hh = lat.width, lat.height
    spots = [(rng.uniform(0, W), rng.uniform(0, Hh), rng.uniform(0.13, 0.30) * max(W, Hh))
             for _ in range(rng.randint(2, 4))]
    if focus:
        pool = list(focus)
        spots = [(c[0] + rng.uniform(-.5, .5), c[1] + rng.uniform(-.5, .5),
                  rng.uniform(0.10, 0.22) * max(W, Hh))
                 for c in rng.sample(pool, min(3, len(pool)))] + spots[:1]

    def weight(wk):
        (x0, y0), (x1, y1) = wk
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        v = 0.12
        for sx, sy, sg in spots:
            v += math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sg * sg))
        return v
    return weight


def _draw_without_replacement(cand, wts, n, rng):
    chosen = set()
    for _ in range(n):
        tot = sum(wts)
        if tot <= 0:
            return None
        r = rng.random() * tot
        acc = 0
        for i, x in enumerate(wts):
            acc += x
            if acc >= r:
                break
        chosen.add(cand[i])
        wts[i] = 0.0
    return chosen


def braid(lat, walls, comps, rng, n_open, focus=None, equal=False):
    """Open exactly n_open closed walls whose two cells are in the SAME
    component, sampled without replacement under the hotspot field.

    `equal=True` splits the budget EVENLY over the components instead.  That is
    what mc needs: its certificate is a ball inside one component, so a high
    band otherwise forces the target's component to be the branchiest one on the
    board, and "answer = the start in the branchiest component" then rises with
    the tier (without equal budgets: 0.217 / 0.383 / 0.833 against a
    0.250 best constant).  Equal budgets make every component equally branchy, so
    that cue has nothing to read."""
    owner = {c: i for i, cc in enumerate(comps) for c in cc}
    cand = [wk for wk in sorted(walls) if owner[wk[0]] == owner[wk[1]]]
    if len(cand) < n_open:
        return None
    w = braid_field(lat, rng, focus)
    if not equal:
        chosen = _draw_without_replacement(cand, [w(x) for x in cand], n_open, rng)
        return None if chosen is None else walls - chosen
    chosen = set()
    per = [n_open // len(comps)] * len(comps)
    for j in range(n_open - sum(per)):
        per[j % len(comps)] += 1
    for i, quota in enumerate(per):
        sub = [x for x in cand if owner[x[0]] == i]
        if len(sub) < quota:
            return None
        got = _draw_without_replacement(sub, [w(x) for x in sub], quota, rng)
        if got is None:
            return None
        chosen |= got
    return walls - chosen


def build_board(lat, rng, K, fracs, focus_random=False, equal=False):
    """-> (walls, comps, carver).  P == len(lat.cells)-1+K exactly."""
    carver = rng.choice(CARVERS)
    tree = lattice.CARVERS[carver](lat.cells, lat.neighbors, rng)
    walls, comps = partition_tree(lat, tree, rng, fracs)
    if walls is None:
        return None
    N, k = len(lat.cells), len(fracs)
    n_open = (N - 1 + K) - (N - k)
    focus = rng.choice(comps) if (focus_random and k > 1) else None
    walls = braid(lat, walls, comps, rng, n_open, focus, equal=equal)
    if walls is None:
        return None
    comps = lattice.components(lat.cells, lat.neighbors, walls)
    if len(comps) != k:
        return None
    return walls, comps, carver


def rand_profile(rng, k, lo=0.12):
    """Shared component-size law.  Random composition with a floor."""
    cuts = sorted(rng.random() for _ in range(k - 1))
    raw = [b - a for a, b in zip([0.0] + cuts, cuts + [1.0])]
    raw = [max(x, lo) for x in raw]
    s = sum(raw)
    return [x / s for x in raw]


# --------------------------------------------------------------------------- #
# Marker separation, shared by the case builders (maze_core, detour_core)
# --------------------------------------------------------------------------- #
def sep_ok(lat, diag, a, b):
    return SEP[0] <= lattice.distance_px(lat, a, b) / diag <= SEP[1]


# --------------------------------------------------------------------------- #
# Rendering + records
# --------------------------------------------------------------------------- #
def render(lat, case, task, rng, palettes, idx, image_rel, tier, K):
    palette = rng.choice(palettes)
    thickness = rng.choice(questions.THICKNESSES)
    style = lattice.make_style(palette, thickness)
    colors = questions.pick_marker_colors(task, rng)
    shape = questions.pick_target_shape(rng) if task != "tf" else None
    pstyle = rng.choice(questions.TARGET_PROMPT_STYLES) if task != "tf" else None
    if task != "tf":
        case["target_shape"] = shape
    markers = questions.markers_for_case(task, case, colors, shape or "square")
    ms = questions.MARKER_SCALE[LATTICE]
    base = lattice.render_case(lat, case["walls"], markers, style=style, marker_scale=ms)
    base, (dx, dy) = lattice.autocrop(base, style.bg)

    desc = (questions.target_descriptor(shape, pstyle) if task != "tf"
            else "the unlabeled target marker")
    if task == "tf":
        q, qt = questions.tf_question(rng)
        rec = questions.tf_record(idx, lat, LATTICE, case["carver"], case, dx, dy,
                          image_rel, question=q, qa_template=qt)
    elif task == "mc":
        q, qt = questions.mc_question(rng, [l for l, _ in case["starts"]], desc)
        rec = questions.mc_record(idx, lat, LATTICE, case["carver"], case, dx, dy,
                          image_rel, question=q, qa_template=qt)
    elif task == "count":
        q, qt = questions.count_question(rng, desc)
        rec = questions.count_record(idx, lat, LATTICE, case["carver"], case, dx, dy,
                             image_rel)
        rec["question"], rec["qa_template"] = q, qt
    else:
        q, qt = questions.closer_question(rng, desc)
        rec = questions.closer_record(idx, lat, LATTICE, case["carver"], case, dx, dy,
                              image_rel)
        rec["question"], rec["qa_template"] = q, qt

    N = len(lat.cells)
    diag = math.hypot(*lat.image_size())
    if task == "tf":
        seps = [lattice.distance_px(lat, case["a"], case["b"]) / diag]
    else:
        seps = [lattice.distance_px(lat, c, case["exit"]) / diag for _l, c in case["starts"]]
    P = len(lattice.all_walls(lat.cells, lat.neighbors)) - len(case["walls"])
    D = sum(1 for c in lat.cells if odeg(lat, case["walls"], c) == 1)
    rec["difficulty"] = tier
    rec["tier"] = tier
    rec["split"] = "eval"
    rec["rendered_image_size"] = list(base.size)
    rec["style"] = {"palette": palette, "palette_source": questions.palette_source(palette),
                    "thickness": thickness, "marker_colors": colors,
                    "target_shape": shape, "target_prompt_style": pstyle,
                    "target_color": colors.get("target") if task != "tf" else None}
    rec["axis"] = {
        "lambda_E": case["lam"], "lambda_glob": lam(excess(lat, case["walls"]), lat.cells),
        "r_star": case["rstar"], "E_cells": case["E"], "E_frac": round(case["E"] / N, 4),
        "K": K, "P": P, "dead_ends": D, "n_components": case["ncomp"],
        "carver": case["carver"], "grid": list(GRID), "N": N,
        "comp_sizes": case.get("comp_sizes"),
        "target_is_largest": case.get("target_is_largest"),
        "euclid_ratio": case.get("euclid_ratio"), "cue_class": case.get("cue_class"),
        "euclid_rank": case.get("euclid_rank"),
        "deg_A": (odeg(lat, case["walls"], case["a"]) if task == "tf" else None),
        "deg_B": (odeg(lat, case["walls"], case["b"]) if task == "tf" else None),
        "min_feature_px_448": 21.9,
        "sep_med": round(st.median(seps), 4), "sep_min": round(min(seps), 4),
    }
    rec["extrapolation"] = {"scenario": f"tier_{tier}", "graph_regime": f"lambda_band_{tier}",
                            "open_rate": None, "sever_rate": None, "transform": "none",
                            "marker_style": "default", "target_shape": shape,
                            "target_prompt_style": pstyle, "downscale": 1.0,
                            "solution_moves": max(0, len(case["path"]) - 1)}
    questions.add_eval_qa(rec)
    return base, rec
