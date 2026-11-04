"""overlap_match -- which candidate outline is one of the shapes in the overlapping stack?

Generator: generator/gen_tiers.py (block-coded signature loci, same-base decoy
hosts, the size law, the shared placement law), with:

  * candidate panels: one registration per item chosen so that all four
    outlines (stroke + 2 px of white) fit their frame and clear the A-D letter;
    a candidate set that cannot fit is refused (gen_tiers.panel_offset);
  * uniqueness margin: every wrong candidate keeps an outline point >= 3.5 px
    from the stack ink pixels at every translation on a 0.5 px lattice, with
    outline samples snapped to pixels (gen_tiers.gap_gate / gap_margins), and
    r* >= 3 (gen_tiers.rstar_of).  Over REAL translations with unsnapped points
    that ink distance can be lower; the verifier's V5/V6
    certify uniqueness over real translations by two other methods;
  * question text: "(A-D) on the right".

One build() = one gen_tiers.make() call: the derived seed goes into make()'s
RNG key, so tiers and splits never share a stream.  make() searches up to 120
candidate families x 140 placements for a board that passes every gate and
returns None otherwise, which becomes a Reject here.
"""
from __future__ import annotations

from visro import api
from visro.seeding import digest, seed_text
from tasks.discrimination_grouping.overlap_match.generator import gen_tiers

TASK = "overlap_match"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"
# fixed namespace of the item key (changing it changes every key)
_KEY_NAMESPACE = seed_text(148159056150355444527912850242424239665)

TIER_KEY = {"easy": "D0", "medium": "D1", "hard": "D2", "xhard": "D3", "xxhard": "D4"}
LETTERS = "ABCD"


def _definition(tier: str) -> str:
    k = TIER_KEY[tier]
    T = gen_tiers.TIERS[k]
    n_fill = T["n"] - 1 - T["H"]
    return (f"gen_tiers.TIERS['{k}']: n={T['n']} stack shapes (target + H={T['H']} same-base decoy hosts + "
            f"{n_fill} fillers), ov={T['ov']}, dpx={T['dpx']} px, vstar={T['vstar']}, "
            f"avis={T['avis']} (B*=1-avis), rec={T['rec']}, maxvis={T['maxvis']}, estar={T['estar']} px, "
            f"chain_run={T['chain_run']} px, cross_min={T['cross_min']}, pfocus={T['pfocus']}, fsim={T['fsim']}; "
            f"shared: 896x504, 4 loci in 2 blocks, R_READ={gen_tiers.R_READ}, E_FLOOR={gen_tiers.E_FLOOR}, "
            f"RUN_FLOOR={gen_tiers.RUN_FLOOR}, DEPTH_FLOOR={gen_tiers.DEPTH_FLOOR}, GAP_FLOOR={gen_tiers.GAP_FLOOR} px "
            f"(0.5 px translation lattice, pixel-snapped samples), "
            f"r*>={gen_tiers.RSTAR_FLOOR}, panel clearance {gen_tiers.PANEL_CLEAR} px")


TIER_DEFINITIONS = {t: _definition(t) for t in api.TIERS}


def _poly(p, off=(0.0, 0.0)):
    """Full float precision on purpose: the verifier snaps outline samples to
    pixels, and a 3-decimal rounding flips the snapping of a few samples per
    candidate, which can move a gap measurement by up to 1 px."""
    return [[float(x + off[0]), float(y + off[1])] for x, y in p]


# ----------------------------------------------------------------- core_key
_D4 = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
       lambda x, y: (-x, y), lambda x, y: (y, x), lambda x, y: (x, -y), lambda x, y: (-y, -x)]


def _canon_poly(pts):
    """Vertex list -> canonical tuple: bbox-centred, 0.5 px grid, smallest
    cyclic rotation over both traversal directions."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    q = [(round((x - cx) * 2) / 2, round((y - cy) * 2) / 2) for x, y in pts]
    n = len(q)
    best = None
    for seq in (q, q[::-1]):
        for s in range(n):
            t = tuple(seq[s:] + seq[:s])
            if best is None or t < best:
                best = t
    return best


def core_key(cand_polys, gold_index) -> str:
    """Content key: the candidate SET (letter order removed) and which member
    is the gold, each outline translation-normalised, canonical over vertex
    order and over the 8 rotations/reflections of the whole set.  The stack is
    deliberately not part of the key: the same candidate set with the same
    gold is the same item whatever the clutter around it."""
    keys = []
    for g in _D4:
        cs = [_canon_poly([g(x, y) for x, y in p]) for p in cand_polys]
        keys.append(repr((sorted(cs), cs[gold_index])))
    return digest([_KEY_NAMESPACE, min(keys)])


# -------------------------------------------------------------------- build
def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    key = TIER_KEY[tier]
    got = gen_tiers.make(int(seed), 0, key, want_rstar=True)
    if got is None:
        raise api.Reject("make: no board passed the gates within 120 families")
    item, pos, m = got
    ans = m["answer"]
    gi = LETTERS.index(ans)
    if item["correct_idx"] != gi:
        raise RuntimeError("gen_tiers: answer letter and correct_idx disagree")
    T = gen_tiers.TIERS[key]
    ox, oy = item["panel_off"]
    anchors = []
    for i in range(4):
        r, c = divmod(i, 2)
        anchors.append([float(gen_tiers.STACK_W + c * gen_tiers.CELL + gen_tiers.PAD + ox), float(r * gen_tiers.CELL + gen_tiers.PAD + oy)])
    shapes = item["shapes"]
    image = gen_tiers.render(item, pos)
    meta = {
        "tier_key": key,
        "answer": ans,
        "gold_index": gi,
        "layout": {"canvas": [gen_tiers.OUT_W, gen_tiers.OUT_H], "stack_w": gen_tiers.STACK_W, "cell": gen_tiers.CELL, "pad": gen_tiers.PAD,
                   "frame": gen_tiers.PANEL_BOX, "frame_line": gen_tiers.FRAME_W, "stroke_px": gen_tiers.STROKE_OUT,
                   "panel_clear_px": gen_tiers.PANEL_CLEAR, "letter_box": list(gen_tiers.LETTER_BOX),
                   "panel_off": [float(ox), float(oy)],
                   "note": "candidate i (A,B,C,D = top-left, top-right, bottom-left, bottom-right) is drawn at "
                           "candidates.polys[i] + candidates.anchors[i]; stack shape k at stack.polys[k] "
                           "(absolute px), drawn in list order (k=0 at the bottom)"},
        "stack": {
            "polys": [_poly(s, pos[k]) for k, s in enumerate(shapes)],
            "pos": [[float(x), float(y)] for x, y in pos],
            "kinds": list(item["kinds"]),
            "target_k": int(item["target_k"]),
            "shape_bbox": gen_tiers.shape_boxes(item, pos),
        },
        "candidates": {
            "polys": [_poly(p) for p in item["cand_polys"]],
            "anchors": anchors,
        },
        "family": {
            "cand_keys": m["cand_keys"], "host_keys": m["host_keys"], "block_map": m["block_map"],
            "correct_key": m["correct_key"], "loci": m["loci"],
        },
        "gates": {
            "gap_margin_px": m["gap_margin_px"], "gap_floor_px": gen_tiers.GAP_FLOOR,
            "rstar_px": m["rstar_px"], "rstar_floor_px": gen_tiers.RSTAR_FLOOR,
            "Estar_px": m["Estar_px"], "emin_all": m["emin_all"], "refute_run_px": m["refute_run_px"],
            "refute_depth_px": m["refute_depth_px"], "r_read_px": gen_tiers.R_READ,
        },
        "difficulty": {
            "n_shapes": T["n"], "n_hosts": T["H"], "Bstar": m["Bstar"], "Vstar": m["Vstar"],
            "arc_vis": m["arc_vis"], "block_runs": m["block_runs"], "occlusion_frac": m["occlusion_frac"],
            "max_vis": m["max_vis"], "occ_rank": m["occ_rank"], "min_ncross": m["min_ncross"],
            "cand_gap_px": m["cand_gap_px"], "host_gap_px": m["host_gap_px"],
        },
        "placement": {"law": m["pos_law"], "dref": m["pos_dref"]},
        "search": {k: m[k] for k in ("geom", "panelfit", "place", "cheap", "solve", "gapfail", "rstarfail")},
    }
    return api.Item(
        question=gen_tiers.QUESTION,
        answer=ans,
        image=image,
        meta=meta,
        core_key=core_key(item["cand_polys"], gi),
        answer_type="choice",
        choices=list(LETTERS),
    )
