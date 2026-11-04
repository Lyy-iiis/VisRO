"""count_points_on_lines -- count the dots each drawn segment passes through.

Generator: generator/gen_tiers.py (gen_one: cpl_build.make_problem + gates
G0..G12) with its siblings cpl_build / cpl_core / cpl_gates / cpl_ink and the
helper generator/base/grid_geometry.py (pt_seg_dist, collinear_grid_points, colours).

Per item:
  * gen_one(tier, derived_seed) builds one candidate and runs the generator's
    own gates G0..G12; a refused candidate raises Reject(<gate name>).
  * The question states the centre rule and the answer format; its format
    example is "2,2;3": two segments, 2 dots each, one shared dot.  No tier can produce any of its numbers -- every segment passes >= 4
    dots, every tier has L >= 3 and so a total >= 9 -- so it shares no count,
    no prefix, no total and no length with any gold.  A gate re-checks that.
  * core_key = sha256 of (task, N, the segment set reduced modulo the 8
    symmetries of the square lattice AND translation).  A rotation, a mirror
    or a whole-picture shift of an accepted item is a duplicate (it has the
    identical answer string), and endpoint order inside a segment and segment
    order do not matter.  Cross-tier collisions are impossible ((N, L) differs
    per tier) but the driver checks anyway.
  * meta carries every number an independent checker needs: the integer
    segment endpoints, the lattice -> pixel map, dot radius, line width,
    overshoot, colours, the tier's acuity constants and the near-miss margin.
"""
from __future__ import annotations

from visro import api
from visro.seeding import digest

TASK = "count_points_on_lines"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

GENERATOR = 'tasks/discrimination_grouping/count_points_on_lines/generator/gen_tiers.py'

# The question's format example is a form no tier can produce (see the module docstring).
EXAMPLE_ANSWER = "2,2;3"
QUESTION = (
    'Blue segments are drawn over a grid of black dots; a segment passes through a dot only if it'
    " crosses its centre. Answer each segment's dot count in descending order, comma-separated, "
    "then ';' and the number of distinct dots covered, no spaces, e.g. "
    + EXAMPLE_ANSWER + "."
)


def _numbers(answer: str):
    counts, total = answer.split(";")
    return [int(x) for x in counts.split(",")], int(total)


def example_overlap(answer: str, example: str = EXAMPLE_ANSWER) -> list[str]:
    """How the format example would give away part of `answer` (empty == it gives away nothing):
    equal strings, the same number of counts, or ANY number in common, in any position
    (this subsumes a shared first count, prefix, pair, total or count vector)."""
    if answer == example:
        return ["equal"]
    (ac, at), (ec, et) = _numbers(answer), _numbers(example)
    out = []
    if len(ac) == len(ec):
        out.append(f"same number of counts ({len(ac)})")
    shared = (set(ac) | {at}) & (set(ec) | {et})
    if shared:
        out.append(f"shares the number(s) {sorted(shared)}")
    return out


_COMMON = ("dots: N x N lattice, canvas margin int(0.08*size), spacing=(size-2*margin)/(N-1), "
           "line width max(2px,0.05*spacing), all segments blue; segments: primitive direction (p,q), "
           "1<=|p|,|q|<=5, |p|!=|q| (no slope 0/inf/+-1), k in [max(3,kmax//2),kmax] steps (>=4 dots), "
           "length >= 0.55*(N-1) spacings, drawn 0.35*spacing past both end dots, on distinct lattice lines; "
           "OFF dots > margin*spacing from every segment body; gates G1 floors (dot dia>=6px, line>=2px, "
           "clearance>=3px), G3-G6 pixel gates, G7 endpoints>=0.8*spacing apart, G8 segments within 15deg "
           ">=1.0*spacing apart, G10 ON-dot clearance>=3px, G11 no slope 0/inf/+-1, G12 per-segment counts "
           "not all equal; answer 'c1,...,cL;total' with counts descending; format example in the question: "
           "'2,2;3' (shares no number with any gold)")
TIER_DEFINITIONS = {
    "easy": "N=11 (121 dots), L=3 segments, canvas 448x448 (spacing 37.80px), dot radius 0.16*spacing "
            "(12.1px dia), near-miss margin U(0.40,0.44)*spacing, >=2 near-miss dots (margin < d <= 1.7*margin), "
            "ON tolerance (G5/G9) 0.18*spacing; " + _COMMON,
    "medium": "N=13 (169 dots), L=4 segments, canvas 448x448 (spacing 31.50px), dot radius 0.12*spacing "
              "(7.6px dia), near-miss margin U(0.28,0.32)*spacing, >=4 near-miss dots, "
              "ON tolerance (G5/G9) 0.15*spacing; " + _COMMON,
    "hard": "N=19 (361 dots), L=12 segments, canvas 672x672 (spacing 31.44px), dot radius 0.10*spacing "
            "(6.3px dia), near-miss margin U(0.24,0.26)*spacing, >=9 near-miss dots, "
            "ON tolerance (G5/G9) 0.13*spacing; " + _COMMON,
    "xhard": "N=20 (400 dots), L=11 segments, canvas 672x672 (spacing 29.79px), dot radius max(3px,0.10*spacing) "
             "(6.0px dia), near-miss margin U(0.24,0.26)*spacing, >=10 near-miss dots, "
             "ON tolerance (G5/G9) 0.13*spacing; above hard in acuity only (finer spacing and dots), "
             "while its workload is lighter than hard's (11 vs 12 segments); " + _COMMON,
    "xxhard": "N=20 (400 dots), L=13 segments, canvas 672x672 (spacing 29.79px), dot radius max(3px,0.10*spacing) "
              "(6.0px dia), near-miss margin U(0.24,0.26)*spacing, >=10 near-miss dots, "
              "ON tolerance (G5/G9) 0.13*spacing; xhard's acuity with 13 segments; " + _COMMON,
}

def canonical_segments(segs, n: int):
    """The segment set reduced modulo the 8 symmetries of the N x N lattice and translation:
    for each symmetry, shift the image so its smallest endpoint x and y are 0, sort the
    endpoints inside each segment and the segments; keep the smallest of the 8 results.
    Translation is quotiented out too because a shifted copy shows the same picture moved
    by whole lattice steps and carries the identical answer."""
    m = int(n) - 1
    ops = (lambda x, y: (x, y), lambda x, y: (m - x, y), lambda x, y: (x, m - y), lambda x, y: (m - x, m - y),
           lambda x, y: (y, x), lambda x, y: (m - y, x), lambda x, y: (y, m - x), lambda x, y: (m - y, m - x))
    best = None
    for op in ops:
        img = [(op(*a), op(*b)) for a, b in segs]
        x0 = min(min(a[0], b[0]) for a, b in img)
        y0 = min(min(a[1], b[1]) for a, b in img)
        img = sorted(tuple(sorted(((a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0)))) for a, b in img)
        if best is None or img < best:
            best = img
    return [[list(a), list(b)] for a, b in best]


def core_key(segs, n: int) -> str:
    return digest([TASK, int(n), canonical_segments([(tuple(a), tuple(b)) for a, b in segs], n)])


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in TIER_DEFINITIONS:
        raise ValueError(f"unknown tier {tier!r}")
    from tasks.discrimination_grouping.count_points_on_lines.generator import cpl_build, cpl_core, gen_tiers
    from tasks.discrimination_grouping.count_points_on_lines.generator.base.grid_geometry import (
        BG, DOT_COLOR, LINE_COLORS)

    cfg = gen_tiers.TIERS[tier]
    ok, why, payload = gen_tiers.gen_one(tier, int(seed))
    if not ok:
        raise api.Reject(why)
    img, m, segs, on_set, diag, to_px, dot_r = payload
    answer = m["answer"]
    if example_overlap(answer):
        raise api.Reject("gold overlaps the question's format example")

    n, size = m["grid_n"], cfg["size"]
    margin_px = int(size * cpl_build.MARGIN_CANVAS)
    spacing = (size - 2 * margin_px) / (n - 1)
    lw = max(cpl_core.MIN_LINE_W_PX, cpl_build.LW_FRAC * spacing)
    assert abs(to_px(1, 1)[0] - (margin_px + spacing)) < 1e-9, "lattice map changed"
    line_rgb = dict(LINE_COLORS)[m["line_color"]]

    meta = {
        "generator": GENERATOR,
        "grid_n": n,
        "n_segs": m["n_segs"],
        "canvas": [size, size],
        "render": {
            "lattice_to_px": "dot (gx,gy) is drawn centred at (margin_px + gx*spacing_px, margin_px + gy*spacing_px); "
                             "gx grows rightwards, gy downwards; drawn at 3x supersampling, LANCZOS-downsampled",
            "margin_px": margin_px,
            "spacing_px": spacing,
            "dot_r_px": dot_r,
            "line_w_px": lw,
            "overshoot_frac": cpl_core.OVERSHOOT,
            "supersample": 3,
            "bg_rgb": list(BG),
            "dot_rgb": list(DOT_COLOR),
            "line_rgb": list(line_rgb),
            "line_color": m["line_color"],
        },
        "tier_params": {
            "dot_r_frac": cfg["dot_r"], "on_tol_frac": cfg["on_tol"],
            "margin_lo": cfg["margin_lo"], "margin_hi": cfg["margin_hi"],
            "nearmiss_min": cfg["nearmiss"], "nearmiss_band": cpl_build.NEARMISS_BAND,
        },
        "segments_grid": [[list(a), list(b)] for a, b in segs],
        "on_dots_grid": [list(p) for p in sorted(on_set)],
        "per_line": list(m["per_line"]),
        "per_line_sorted": list(m["per_line_sorted"]),
        "n_on": m["n_on"],
        "sum_per_line": m["sum_per_line"],
        "n_shared_pts": m["n_shared_pts"],
        "prim_dirs": [list(d) for d in m["prim_dirs"]],
        "margin_frac": m["margin_frac"],
        "near_margin_px": m["near_margin_px"],
        "n_nearmiss": m["n_nearmiss"],
        "gen_gates": {
            "clearance_px": m["clearance_px"],
            "on_clear_px": diag["on_clear_px"],
            "min_white_gap_px": diag["min_white_gap"],
            "min_endpoint_sep_px": m["min_endpoint_sep_px"],
            "min_parallel_sep_px": m["min_parallel_sep_px"],
            "rings": diag["rings"],
        },
    }
    return api.Item(
        question=QUESTION,
        answer=answer,
        image=img,
        meta=meta,
        core_key=core_key(segs, n),
        answer_type="text",
    )
