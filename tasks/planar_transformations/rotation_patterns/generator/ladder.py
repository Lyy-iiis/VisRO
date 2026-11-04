# -*- coding: utf-8 -*-
"""rotation_patterns: the FIVE-TIER ladder, one TIERS table, one generator.

All five tiers are rows of ONE table below and come out of ONE build().

    tier    L  pool                       layout  min_gap  questions
    easy    3  _pool_L3w8full (1 in 16)   strip6  2        2 absolute
            4  _pool_L4w8full (15 in 16)
    medium  5  _pool_L5m                  strip6  1        3 absolute
    hard    5  _pool_L5w8                 wide12  1        9 absolute
    xhard   5  _pool_L5xwd                wide12  1        10 absolute + 2 none
    xxhard  5  _pool_L5w8                 wide12  1        7 absolute + 2 none
                                                           + 3 chains at depth 1

The pools (word_pools.py; built by build/exhaustive_pools.py and
build/length5_pools.py)
keep every word gate W1..W8, W5 included, and the sampler is not restricted to
one level multiset, which is what gives the top tiers a supply of tens of
thousands of figures.

XHARD'S TWIN FAMILY
-------------------
d(8), the number of spokes between a figure and its 180 deg twin, is a set:
d(8) in {6, 8, 10}.  Nothing the reader sees depends on it - same wide12 canvas
of 1232x364, same 12 candidates, same 10 absolute + 2 none questions, same L=5
with its 8.25 px bead step, same min_gap=1 - and the property that carries the
tier survives, because W5's floor is tied to each word's own d(8): the closest
pair of rings on the canvas is STILL the 180 deg twin, for every item.  No
SINGLE d(8) reaches 25,000 figures (build/length5_pools.py has the table).

EASY DRAWS FROM TWO POOLS AT ONCE
---------------------------------
L=3 is the whole reason easy is easy - 16.50 delivered px between adjacent bead
radii, against 11.00 at L=4 and 8.25 at L=5 - and it is also the reason easy
cannot be built at full size from L=3 alone: the ENTIRE L=3 family that
clears W1..W8 is 1,553 words, i.e. 2,264 distinct figures, and no figure may be
shown twice.  25,000 items is eleven times that.

So easy draws from BOTH pools.  The row keeps L=3 and _pool_L3w8full.json as
its primary source and gains a `mix` entry naming the second one; ladder.source()
decides which of the two an item comes from:

    source(tier, idx) = (L, pool)   for tier "easy":
        idx % 16 == 0   ->  (3, _pool_L3w8full.json)     1 item in 16
        otherwise       ->  (4, _pool_L4w8full.json)    15 items in 16

  DETERMINISTIC.  source() is a pure function of (tier, idx).  No rng is
      consulted - build()'s rng is constructed AFTER the source is fixed and its
      key (rngkey:seed:idx) does not move - so the same idx lands on the same
      pool in every run, on every machine, before a single word is drawn.
  DECLARABLE.  build() returns the pool it actually used.  The question stem
      "it takes one of {L} lengths, and the {L} faint grey circles mark them"
      is filled from the SAME L that drew the word and drew the pixels, because
      question_text is called with the local L, not with TIERS[tier]["L"].
  SELF-LIMITING.  1 in 16 is 1,563 L=3 items in 25,000 (indices 0..24999 with
      idx % 16 == 0), which is 69% of the 2,264-figure ceiling.  If the L=3 half
      ever does run dry, build() returns None on those indices and the run
      simply carries on with L=4 instead of stalling - the ratio degrades, the
      tier still delivers.

THE TWO LOW TIERS
-----------------
easy and medium scale three things against the upper tiers: the number of
asked rotations, the coarseness of the angle set, and how near-symmetric the
figure is.

  * a bank of 6               -> slots = 6, a strip canvas of 1232x224 (352
                                visual tokens) with the SAME 168 px figure box,
                                so a figure is exactly as legible as at the top
                                of the ladder.
  * 2 / 3 asked rotations     -> n_abs = 2 / 3.
  * coarse vs fine angle set  -> min_gap.  min_gap=2 forces every pair among
                                {base} + the 6 candidates to be at least 45 deg
                                apart; min_gap=1 lets the bank use the full
                                22.5 deg lattice, so some pairs sit one notch
                                apart.  45 deg is the coarsest step that still
                                fills a 6-slot bank once 0 deg and 180 deg are
                                excluded.
  * near-symmetry             -> L, the number of bead lengths, which is what
                                "how much does this figure resemble itself"
                                means for this symbol: L=3 puts 16.50 delivered
                                px between adjacent bead radii, L=4 puts 11.00,
                                L=5 puts 8.25 (the legibility floor the three
                                upper tiers sit on).

The L=3 and L=4 pools are the gate stack W1..W8 enumerated exhaustively over
all 3^16 and 4^16 words (build/exhaustive_pools.py), without W9, the exact kill of
the second ink multipole: W9's family leaves only 7 words at L=3 and 40 at L=4,
against 1,553 and 34,811 without it.  It costs the m=2 probe, which rises to its
own ceiling (~0.8 per question on a 6-slot bank).  W8 and every combinatorial
gate, W5 included, apply unchanged.

The bank is rng.sample'd out of the 14 legal shifts and then shuffled again
(S1/S2), so a text-only solver is at chance on every tier.

gate stack (identical for all five tiers; the low tiers add one and drop none)
  S1 slot + question shuffle          S2 bank is a proper subset of the 14
  S4 exact 16-fold lattice            G1 worst_cc >= 110
  G3 exhaustive complete solver       G4 primitivity
  G5 rotation fidelity                G6 count entropy 0
  G8 level legibility on delivered px G11 badge does not touch the dial
  G12/G13/G15..G18 in question.py     W1..W8 word gates
  G19 min_gap, the bank's own angular separation floor (>=1 = no floor)
The centroid probe (m=1) is recorded in meta, not gated; W8 on the pools guards
the centroid shortcut.
"""
from __future__ import annotations
from visro.seeding import seed_text
import random
import functools
import itertools
import numpy as np
from PIL import Image, ImageDraw

from . import word as W, dial as D, moment as MO, answer_probe as PR, question as Q, word_pools as WP
from . import geometry_gates as B      # geometry gate constants, MIN_CC / MAX_ROTDIFF / ...
from .word import K

# ------------------------------------------------------------------ layouts
# Both put the base in column 0 and the candidates in the columns after it, and
# both resolve to box = 168, the figure size of every tier.
LAYOUTS = {
    # the canvas of the three top tiers: 1232x364 = 572 tokens
    "wide12": dict(CW=1232, CH=364, COLS=7, ROWS=2, M=10, GAP=6, slots=12),
    # one row of 6 candidates + the base: 1232x224 = 352 tokens
    "strip6": dict(CW=1232, CH=224, COLS=7, ROWS=1, M=10, GAP=6, slots=6),
}

# easy's mixing ratio: one item in MIX_PERIOD comes from the primary (L=3)
# source and the rest from the mix (L=4).  16 puts 1,563 L=3 items in a
# 25,000-item tier, 69% of the 2,264 figures L=3 has in all.  This is the ONE
# number that sets the ratio, and it is the one to raise if the L=3 half turns
# out not to reach 69% of its ceiling under the geometry gates: 32 asks for 782
# figures (35%), 8 asks for 3,125 and is not available at all.
MIX_PERIOD = 16

TIERS = {
    # TWO SOURCES, see the header: the full 1,553-word L=3 enumeration, and on the
    # indices it does not take, the full L=4 enumeration (`mix`).  MIX_PERIOD is
    # the one number that sets the ratio.
    "easy":   dict(L=3, pool="_pool_L3w8full.json", lay="strip6", min_gap=2,
                   n_abs=2,  n_none=0, chains=[], rngkey=seed_text(2110959926963322581881),
                   mix=dict(L=4, pool="_pool_L4w8full.json",
                            period=MIX_PERIOD, primary_at=(0,))),
    # medium draws from hard's own L=5 word family, ENUMERATED (not sampled), with
    # W8 at 0.17 px instead of 0.20 for this tier only (length5_pools.build_medium
    # has the why and the numbers).  easy is the only tier drawing from
    # _pool_L4w8full.
    "medium": dict(L=5, pool="_pool_L5m.json", lay="strip6", min_gap=1,
                   n_abs=3,  n_none=0, chains=[], rngkey=seed_text(138343869773477121746367853)),
    # ---- the top three tiers ----------------------------------------------
    "hard":   dict(L=5, pool="_pool_L5w8.json", lay="wide12", min_gap=1,
                   n_abs=9,  n_none=0, chains=[], rngkey=seed_text(2321024981777199683260863486194276)),
    "xhard":  dict(L=5, pool="_pool_L5xwd.json", lay="wide12", min_gap=1,
                   n_abs=10, n_none=2, chains=[], rngkey=seed_text(2321024981777417570856705112240740)),
    "xxhard": dict(L=5, pool="_pool_L5w8.json", lay="wide12", min_gap=1,
                   n_abs=7,  n_none=2, chains=[(1, False)] * 3,
                   rngkey=seed_text(594182395334963118914781121301541476)),
}


def source(tier, idx):
    """Which (L, pool) item `idx` of `tier` draws from.  A PURE FUNCTION of
    (tier, idx): no rng, no state, no dependence on what has already been
    emitted, so the same index lands on the same source in every run.

    A tier without a `mix` entry - medium, hard, xhard, xxhard - returns its own
    row unchanged, which is why this can be threaded through build() without
    moving one byte of theirs."""
    spec = TIERS[tier]
    mx = spec.get("mix")
    if mx is None or (idx % mx["period"]) in mx["primary_at"]:
        return spec["L"], spec["pool"]
    return mx["L"], mx["pool"]


def n_questions(tier):
    t = TIERS[tier]
    return t["n_abs"] + t["n_none"] + len(t["chains"])


@functools.lru_cache(maxsize=None)
def pool(name):
    d = WP.load(name)
    return [tuple(x) for x in d["words"]], d["cfg"]


# ------------------------------------------------------------------- layout
def layout(lay):
    p = LAYOUTS[lay]
    box = int(min((p["CH"] - 2 * p["M"] - (p["ROWS"] - 1) * p["GAP"]) / p["ROWS"],
                  (p["CW"] - 2 * p["M"] - (p["COLS"] - 1) * p["GAP"]) / p["COLS"]))
    y0 = p["M"] + (p["CH"] - 2 * p["M"]
                   - (p["ROWS"] * box + (p["ROWS"] - 1) * p["GAP"])) // 2
    return box, p["M"], y0


def base_origin(lay):
    box, x0, y0 = layout(lay)
    return x0, y0


def tile_origin(lay, i):
    p = LAYOUTS[lay]
    box, x0, y0 = layout(lay)
    c, r = i % (p["COLS"] - 1) + 1, i // (p["COLS"] - 1)
    return x0 + c * (box + p["GAP"]), y0 + r * (box + p["GAP"])


def _colour_mask(img, x, y, box):
    a = np.asarray(img.crop((x, y, x + box, y + box))).astype(np.int16)
    return (a.max(axis=2) - a.min(axis=2)) > 30


def _badge(d, x, y, text, font):
    tb = d.textbbox((0, 0), text, font=font)
    w, h = tb[2] - tb[0], tb[3] - tb[1]
    pad = 2
    d.rectangle([x + 1, y + 1, x + 1 + w + 2 * pad, y + 1 + h + 2 * pad],
                fill=(255, 255, 255), outline=(120, 120, 128), width=1)
    d.text((x + 1 + pad - tb[0], y + 1 + pad - tb[1]), text, font=font, fill=(20, 20, 20))


def compose(lay, base_img, tiles):
    """The candidate canvas: the base figure and the numbered candidate tiles of layout `lay`."""
    p = LAYOUTS[lay]
    box, x0, y0 = layout(lay)
    assert box == D.BOX, (box, D.BOX)
    img = Image.new("RGB", (p["CW"], p["CH"]), D.BG)
    d = ImageDraw.Draw(img)
    nf, bf = D.font_num(13), D.font_num(15)
    bx, by = base_origin(lay)
    d.rectangle([bx - 4, by - 4, bx + box + 4, by + box + 4], outline=(45, 45, 45), width=4)
    t = base_img.resize((box, box), Image.LANCZOS)
    img.paste(t, (bx, by), t)
    lb = d.textbbox((bx + 2, by + box + 10), "base", font=bf)
    assert lb[3] < p["CH"] and lb[2] < p["CW"], f"{lay}: the base label is clipped"
    d.text((bx + 2, by + box + 10), "base", font=bf, fill=(35, 35, 35))
    before = []
    for i, tl in enumerate(tiles):
        x, y = tile_origin(lay, i)
        d.rectangle([x - 3, y - 3, x + box + 3, y + box + 3], outline=(200, 200, 205), width=2)
        tt = tl.resize((box, box), Image.LANCZOS)
        img.paste(tt, (x, y), tt)
        before.append(_colour_mask(img, x, y, box).sum())
    for i in range(len(tiles)):
        _badge(d, *tile_origin(lay, i), str(i + 1), nf)
    for i in range(len(tiles)):                                    # G11
        x, y = tile_origin(lay, i)
        assert _colour_mask(img, x, y, box).sum() == before[i], \
            f"index badge overlaps the dial of figure {i+1}"
    return img


# ------------------------------------------------------------- G19: the bank
@functools.lru_cache(maxsize=None)
def banks(slots, min_gap):
    """Every legal bank whose figures - and the base with them - are at least
    min_gap notches apart.  Enumerated, not searched, so the draw is uniform."""
    cand = Q.legal_shifts()
    out = []
    for c in itertools.combinations(cand, slots):
        pts = (0,) + c
        if all(min((a - b) % K, (b - a) % K) >= min_gap
               for a, b in itertools.combinations(pts, 2)):
            out.append(c)
    return out


def sample_bank(rng, slots, min_gap):
    cand = Q.legal_shifts()
    assert slots < len(cand), "S2: bank must be a proper subset"
    if min_gap <= 1:
        sh = rng.sample(cand, slots)                 # the measured tiers' call
        rng.shuffle(sh)                              # S1
        return sh
    bk = banks(slots, min_gap)
    assert bk, f"no bank of {slots} slots has min gap {min_gap}"
    sh = list(bk[rng.randrange(len(bk))])
    rng.shuffle(sh)                                  # S1
    return sh


def depends_on(plan, shifts, slots):
    """How many distinct figures must be decoded before the item is safe.  A
    plain question needs its own target figure; a chain needs its root and every
    intermediate too; a NONE question needs every one of them, because 'nothing
    matches' is only established by checking all."""
    if any(s["absent"] for s in plan):
        return slots
    dep = set()
    for s in plan:
        for t in Q.path_shifts(s, shifts):
            dep.add(shifts.index(t))
    return len(dep)


def build(seed, idx, tier):
    spec = TIERS[tier]
    # The source is fixed BEFORE the rng exists, and from (tier, idx) alone.
    # For every tier but easy this is exactly spec["L"], spec["pool"].
    L, poolname = source(tier, idx)
    lay = spec["lay"]
    slots = LAYOUTS[lay]["slots"]
    P, cfg = pool(poolname)
    assert P, f"{tier}: empty pool"
    creq = cfg.get("coarse_req", 3)
    rng = random.Random(f"{spec['rngkey']}:{seed}:{idx}")
    for attempt in range(600):
        w0 = P[rng.randrange(len(P))]
        w = rng.choice(W.variants(w0, L))
        assert W.combinatorial_ok(w, L, cfg["m_req"], cfg["run_cap"],
                                  cfg["min_ham"], creq)
        col = D.PALETTE[rng.randrange(len(D.PALETTE))]
        B.geometry_gates(L, col)                                    # G8

        shifts = sample_bank(rng, slots, spec["min_gap"])           # S1 S2 G19
        plan = Q.make_plan(rng, shifts, spec["n_abs"], spec["n_none"],
                           spec["chains"], slots)                   # S1
        assert len(plan) == n_questions(tier)

        tg = [Q.target_of(s, shifts) for s in plan]
        assert len(set(tg)) == len(tg), "G12: two questions share a target"
        assert all(t not in (0, K // 2) for t in tg), "G13: free none"
        n_pres = sum(not s["absent"] for s in plan)
        assert n_pres <= Q.max_present(slots), f"G18: {n_pres} present answers"
        chains = [s for s in plan if s["kind"] == "chain"]
        refs = [s["ref"] for s in chains]
        assert len(set(refs)) == len(refs), "G17: two chains share a root"
        assert all(1 <= r <= slots for r in refs), "G17: bad root"

        ansl, words = Q.solve_plan(w, shifts, plan)                 # G3 G4 G13 G15 G16
        assert len({tuple(sorted(cw)) for cw in words + [tuple(w)]}) == 1   # G6
        n_none = sum(s["absent"] for s in plan)
        assert sum(a == "none" for a in ansl) == n_none
        assert len({a for a in ansl if a != "none"}) == n_pres, "answers collide"

        base_img = D.render_tile(tuple(w), L, col)
        tiles = [D.render_tile(cw, L, col) for cw in words]
        base_arr = D.flatten(base_img)
        arrs = [D.flatten(t) for t in tiles]
        # the centroid probe is recorded, not gated: W8 on the pool guards the
        # centroid shortcut
        m1 = PR.probe_hits_arr(base_arr, arrs, shifts, 1)
        cc = D.worst_cc(arrs + [base_arr], floor=B.MIN_CC)          # G1
        if cc < B.MIN_CC:
            continue
        rf = D.rot_fidelity(base_img, arrs, shifts)                 # G5
        if rf > B.MAX_ROTDIFF:
            continue
        m2 = PR.probe_hits_arr(base_arr, arrs, shifts, 2)
        m3 = PR.probe_hits_arr(base_arr, arrs, shifts, 3)
        cent = D.centroid_of(D.color_ink(base_arr))[0]

        qtext = Q.question_text(L, plan, shifts, slots)
        ans = ",".join(ansl)
        assert qtext.isascii() and ans.isascii()
        for s in chains:                                            # G17
            # rotation_patterns prints a chain as its own line "figure k+A"
            assert f"\nfigure {s['ref']}+" in qtext, "root not in the stem"
        return dict(q=compose(lay, base_img, tiles), L=L, lay=lay,
                    pool=poolname,
                    slots=slots, min_gap=spec["min_gap"],
                    w=tuple(w), shifts=shifts, plan=plan, ansl=ansl,
                    ans=ans, qtext=qtext, col=col, cc=cc, rotdiff=rf,
                    n_q=len(plan), n_none=n_none, n_present=n_pres,
                    n_chain=len(chains), dep=depends_on(plan, shifts, slots),
                    m1=m1, m2=m2, m3=m3, cent_meas=cent,
                    cent_pred=MO.centroid_px(w, L),
                    m=W.m_anchor(w), coarse=W.coarse_anchor(w, L),
                    minham=W.min_hamming(w),
                    tries=attempt + 1, step=D.level_step(L))
    return None
