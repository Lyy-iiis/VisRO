"""The sampler of the natural-rival rungs (hard / xhard / xxhard): a class-balanced
draw over the orientation-augmented rival graph, the law's pre-screen, and
puzzle.build_shadows with two early stops that change no item.

A natural-rival item's key is its set of base silhouettes, drawn as disjoint
natural-rival edges of a graph.  Over the upright icons alone (59 / 42 / 36
edges over 77 / 60 / 51 icons) that graph has fewer keys than the cells need,
so it is enlarged along the one axis that changes what a reader sees without
touching any difficulty knob -- the orientation of the base icons -- and drawn
from so that a few hub icons do not dominate:
  * nodes: every drawable silhouette row in each of the 8 orientations of the
    square (puzzle.ORIENT_OPS; an orientation that reproduces another one of the
    same icon within the twin rule is dropped; text glyphs -- letters, digits,
    brand marks, playing cards -- upright only); the gate tables M / OUT / RAD
    are the cache's on the upright block and computed the same way
    (504 px delivery render, alpha > 40, norm_mask 160 -> 92) on the others;
  * strikes: build/build_strikes.py's census (K = 4 trials of the node's family
    alone, strike on the first failure) run PER NODE (icon, orientation),
    because clone synthesis is not orientation-invariant: upright rows keep
    strikes_92.json, and every other orientation is judged by its own trials
    (an icon struck upright may be drawn rotated or mirrored when that node
    passes);
  * per-rung exclusions (build/build_law.py TIER_DROP, to keep each rung at its
    difficulty level): xhard leaves out the two trophy rival pairs (they drive
    the area shortcut), xxhard the rescued classes that read harder and the
    account-circle / circle-half-full pair;
  * ALLOW additionally forbids two nodes of one identity class and two nodes
    that a 90/180/270 rotation or mirror would bring within 3.5 px (what the
    verifier rejects as "shadow is pattern rotated or mirrored");
  * selection: puzzle.select_base's disjoint-edge construction with the next
    edge drawn class-balanced -- a uniform identity class among those that
    still have a feasible edge, a uniform partner class, then a uniform
    orientation pair -- instead of a uniform edge (select_balanced);
  * key: the sorted IDENTITY classes of the base icons: icons whose nodes are
    twins (IoU >= 0.9, XOR disc <= 2.5 px; merges the playing-card near-twins)
    in any orientation are one class, and the orientation is ignored, so one
    icon set is one item whatever orientations it was drawn in.
All of it is precomputed by generator/build/build_law.py into
CACHE_DIR/law/{law.json, sel_<tier>.npz}; VISRO_FTS_LAW overrides the
directory.
"""
import math

import numpy as np
import cv2
from PIL import Image

from . import decoys, puzzle
from .base import silhouette

# Cross-family early abort: 3 in 4 failed attempts fail the >= 8 px refutation gate, and 9 in 10 of those
# because a clone of one family came within 8 px of ANOTHER family's base or clone -- which is already visible
# on the synthesized masks.  build_shadows_law gives up an attempt as soon as an accepted clone is within XDISC
# px (IoU >= XIOU) of another family's silhouette on the normalised synthesis masks, instead of finishing the
# synthesis, rendering and auditing an image the gate then rejects.  The margins absorb the PNG-vs-synthesis
# difference, so the rule removes work, not items.
XDISC, XIOU = 7.5, 0.64
# Exact pre-audit: puzzle.audit judges the RENDERED PNG, but every tile of that PNG is a pure function of what
# is already known while the clones are synthesized -- a shadow's tile of its synthesis mask and the shadow
# colour, a pattern's tile of its icon, orientation and tint (fit_tile into its own 92 px box on white; nothing
# else is drawn inside a tile box).  PreAudit re-derives those tiles' masks exactly as png_masks does and
# evaluates, pair by pair as shadows appear, the audit gates that one pair or one letter decides (refutation
# disc, family disc, self IoU, true-pair disc, clone band, polarity).  It gives an attempt up only when such a
# gate fails for EVERY render draw still possible (each of the 4 shadow colours; for a pattern, each of the 10
# tints) -- i.e. when the audit is certain to reject the image -- so it removes work, never an item.


def select_balanced(T, avail, nbase, dp, rng, cls, tries=400):
    """puzzle.select_base with the next edge drawn class-balanced: a uniform identity class among the
    classes that still have a feasible edge, a uniform partner class among its feasible partners, then a
    uniform feasible orientation pair between the two.  Feasibility (ALLOW), the same-outline budget
    (sb_max), the odd extra rival and the final role shuffle are select_base's."""
    ALLOW, SB, BAND, E, _ = T
    n = ALLOW.shape[0]
    cE = cls[E]
    for _ in range(tries):
        feas = avail.copy()
        sbc = np.zeros(n, np.int32)
        chosen = []
        while len(chosen) + 2 <= nbase:
            idx = np.flatnonzero(feas[E[:, 0]] & feas[E[:, 1]])
            if len(idx) == 0:
                break
            ce = cE[idx]
            classes = np.unique(ce)
            c1 = classes[rng.randrange(len(classes))]
            sub = idx[(ce[:, 0] == c1) | (ce[:, 1] == c1)]
            partners = np.unique(np.where(cE[sub, 0] == c1, cE[sub, 1], cE[sub, 0]))
            c2 = partners[rng.randrange(len(partners))]
            sub = sub[((cE[sub, 0] == c1) & (cE[sub, 1] == c2)) | ((cE[sub, 1] == c1) & (cE[sub, 0] == c2))]
            a, b = (int(x) for x in E[sub[rng.randrange(len(sub))]])
            chosen += [a, b]
            feas &= ALLOW[a] & ALLOW[b]
            sbc = sbc + SB[a] + SB[b]
            feas &= sbc <= dp["sb_max"]
            feas[[a, b]] = False
        if len(chosen) == nbase - 1:                    # odd nbase: one extra rival
            c = np.flatnonzero(feas & BAND[chosen].any(0))
            if len(c) == 0:
                continue
            chosen.append(int(c[rng.randrange(len(c))]))
        if len(chosen) != nbase:
            continue
        rng.shuffle(chosen)
        return chosen


def xconflict(cn, b, base_n, done):
    """A clone of family b (normalised mask cn) within XDISC px (IoU >= XIOU) of another family's base or of
    a clone of another family accepted so far."""
    for j, bn in enumerate(base_n):
        if j != b and decoys.iou(cn, bn) >= XIOU and decoys.disc_radius(cn, bn) < XDISC:
            return True
    for fb, fn in done:
        if fb != b and decoys.iou(cn, fn) >= XIOU and decoys.disc_radius(cn, fn) < XDISC:
            return True
    return False


class PreAudit:
    """Exact early abort for one attempt (see the pre-audit note above).  Tiles are judged under a probe draw
    (shadow colour 0, tint 0) as they appear; a probe failure is re-judged under every colour still open (for a
    letter: under every tint), and the attempt is doomed once every colour is -- by a shadow pair (refutation
    disc with IoU >= 0.62 across families, any disc inside a family) or by a letter that fails under all tints
    (self IoU, true-pair disc, refutation disc, clone band).  Each condition is the audit's own, with its
    rounding, on the tile masks png_masks would extract, so a doomed attempt is one the audit rejects for every
    render draw."""

    PROBE_C, PROBE_T = 0, 0

    def __init__(self, lib, ents, refs, dp):
        self.lib, self.ents, self.refs, self.dp = lib, ents, refs, dp
        self.L, self.TILE, self.S = dp["L"], dp["TILE"], dp["TILE"] * puzzle.SS
        self.cols = list(silhouette.SHADOW_COLORS)
        self.tints = list(silhouette.SPECIFIC_TINTS)
        A = puzzle.ACCEPT
        self.self_min, self.true_max, self.disc_min = A["self_iou_min"], A["true_disc_max"], A["disc_min"]
        self.lo, self.hi = dp["clone_lo"], dp["clone_hi"]
        self.masks, self.fam = [], []
        self.Bm, self.Am = {}, {}
        self.doomed = set()            # shadow-colour indices under which the audit is certain to reject
        self.reason = None

    # -- tile masks exactly as puzzle.png_masks extracts them from the canvas
    def _png(self, tile):
        T = self.TILE
        can = Image.new("RGB", (T, T), silhouette.BG)
        silhouette.fit_tile(can, (0, 0, T, T), tile)
        sub = puzzle.ink_mask(np.asarray(can.convert("L")))
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
        if n > 1:
            sub = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        return decoys.norm_mask(sub, T)

    def B(self, j, c):
        v = self.Bm.get((j, c))
        if v is None:
            v = self.Bm[(j, c)] = self._png(puzzle.shadow_rgba(self.masks[j], self.cols[c]))
        return v

    def A(self, i, t):
        v = self.Am.get((i, t))
        if v is None:
            v = self.Am[(i, t)] = self._png(puzzle.base_tile(self.lib, self.ents, self.refs[i], self.S,
                                                             self.tints[t]))
        return v

    # -- the audit's conditions (puzzle.audit + the adapter's gates, same rounding)
    def _pair_fail(self, x, y, c):
        bx, by = self.B(x, c), self.B(y, c)
        if self.fam[x] != self.fam[y] and decoys.iou(bx, by) < 0.62:
            return None
        if round(float(decoys.disc_radius(bx, by)), 2) < self.disc_min:
            return "clone_pair_disc" if self.fam[x] == self.fam[y] else "disc"
        return None

    def _letter_fail(self, i, t, c, js):
        a = self.A(i, t)
        for j in js:
            b = self.B(j, c)
            v = decoys.iou(a, b)
            if j == i:
                if round(float(v), 4) < self.self_min:
                    return "self_iou"
                if round(float(decoys.disc_radius(a, b)), 2) > self.true_max:
                    return "true_flat"
                continue
            if round(float(v), 4) > self.hi:
                return "clone_band_hi"
            if self.fam[j] == i and round(float(v), 4) < self.lo:
                return "clone_band_lo"
            if v >= 0.62 and round(float(decoys.disc_radius(a, b)), 2) < self.disc_min:
                return "disc"
        return None

    def _open(self):
        return [c for c in range(len(self.cols)) if c not in self.doomed]

    def _doom(self, cs, why):
        if cs:
            self.doomed |= set(cs)
            if self.reason is None:
                self.reason = why
        return len(self.doomed) == len(self.cols)

    def _judge_pair(self, x, y):
        cs = [c for c in self._open() if self._pair_fail(x, y, c)]
        return self._doom(cs, self._pair_fail(x, y, self.PROBE_C) or "disc")

    def _judge_letter(self, i, why):
        js = list(range(len(self.masks)))
        cs = []
        for c in self._open():
            if all(self._letter_fail(i, t, c, js) for t in range(len(self.tints))):
                cs.append(c)
        return self._doom(cs, why)

    # -- driving: the true shadows first, then each accepted clone
    def start(self, base_masks):
        self.masks = list(base_masks)
        self.fam = list(range(len(base_masks)))
        n = len(self.masks)
        for i in range(self.L):
            w = self._letter_fail(i, self.PROBE_T, self.PROBE_C, range(n))
            if w and self._judge_letter(i, w):
                return True
        for x in range(n):
            for y in range(x + 1, n):
                if self._pair_fail(x, y, self.PROBE_C) and self._judge_pair(x, y):
                    return True
        return False

    def add(self, mask, family):
        self.masks.append(mask)
        self.fam.append(family)
        j = len(self.masks) - 1
        for x in range(j):
            if self._pair_fail(x, j, self.PROBE_C) and self._judge_pair(x, j):
                return True
        for i in range(self.L):
            w = self._letter_fail(i, self.PROBE_T, self.PROBE_C, [j])
            if w and self._judge_letter(i, w):
                return True
        return False


def polarity_bad(dp, meta, families=None):
    """The adapter gates' polarity block (needs only the synthesis meta): the first failing polarity gate or
    None.  families: judge only family_polarity of these letter families (a family is final once synthesized)."""
    L, C = dp["L"], dp["C"]
    if families is not None:
        if C >= 2:
            for i in families:
                if i < L:
                    fp = [m["polarity"] for m in meta if m["kind"] == "clone" and m["src"] == i]
                    if min(fp.count("smoother"), fp.count("extra")) < 1:
                        return "family_polarity"
        return None
    pol = [m["polarity"] for m in meta if m["kind"] == "clone"]
    if min(pol.count("smoother"), pol.count("extra")) < 2:
        return "polarity_mix"
    if C >= 2:
        for i in range(L):
            fp = [m["polarity"] for m in meta if m["kind"] == "clone" and m["src"] == i]
            if min(fp.count("smoother"), fp.count("extra")) < 1:
                return "family_polarity"
    lp = [m["polarity"] for m in meta if m["kind"] == "clone" and m["src"] < L]
    if min(lp.count("smoother"), lp.count("extra")) < 1:
        return "letter_polarity_mix"
    return None


def build_shadows_law(lib, ents, idxs, dp, rng, why, pa):
    """puzzle.build_shadows, line for line, plus two early stops that change no item: the cross-family abort
    (XDISC / XIOU: None with why["_xdisc"]) and the exact pre-audit (pa, a PreAudit: None with why["_pre"] = the
    gate it proves will fail).  The random stream is consumed exactly as in build_shadows up to the stop."""
    L, S, TILE = dp["L"], dp["TILE"] * puzzle.SS, dp["TILE"]
    base_masks, tgt = [], []
    for k in idxs:
        img = puzzle.base_tile(lib, ents, k, S, (0, 0, 0))
        m = np.asarray(img)[..., 3] > 127
        base_masks.append(m)
        area = float(decoys.norm_mask(m, TILE).sum())
        tgt.append(float(np.clip(1.0 - np.pi * puzzle.R_TARGET ** 2 / max(area, 1.0),
                                 dp["clone_lo"] + 0.015, dp["clone_hi"] - 0.005)))
    if pa.start(base_masks):
        why["_pre"] = pa.reason
        return None
    base_n = [decoys.norm_mask(m, TILE) for m in base_masks]
    other = {"smoother": "extra", "extra": "smoother"}
    plan = [(i, c) for i, c in enumerate(dp["letter_clones"])] + \
           [(L + i, c) for i, c in enumerate(dp["u_clones"])]
    masks = list(base_masks)
    meta = [dict(kind="real", src=j) for j in range(len(base_masks))]
    order = [(b, c) for b, c in plan if c]
    rng.shuffle(order)
    done = []                                   # (family, normalised clone) accepted so far
    for b, c in order:
        pols = puzzle.polarity_plan(c, rng)
        used, fam_n, blocked = [], [], []
        for ci in range(c):
            want = pols[ci]
            got = None
            cands = ([want] * 6 if ci < 2 else
                     [want, want, want, other[want], other[want], other[want]])
            for att, pp in enumerate(cands):
                shrink = (1.0, 1.0, 0.86, 0.86, 0.72, 0.62)[min(att, 5)]
                cand = decoys.make_decoy(
                    base_masks[b], tgt[b], TILE, dp["rmin"], dp["rmax"], rng,
                    polarity=pp, why=why, avoid=used + blocked,
                    avoid_r=dp["avoid_frac"] * shrink * S)
                if cand is None:
                    continue
                cn = decoys.norm_mask(cand[0], TILE)
                if any(decoys.disc_radius(cn, f) < dp["rmin"] + 0.5 for f in fam_n):
                    blocked.append(tuple(cand[1]["site"]))
                    puzzle.tally(why, "clone_too_close")
                    continue
                got = cand; pols[ci] = pp
                break
            if got is None:
                puzzle.tally(why, "fail_base")
                return None
            m2, info = got
            # ---- cross-family early abort (the refutation gate would reject this image)
            if xconflict(cn, b, base_n, done):
                why["_xdisc"] = 1
                return None
            done.append((b, cn))
            used.append(tuple(info["site"]))
            fam_n.append(decoys.norm_mask(m2, TILE))
            masks.append(m2)
            meta.append(dict(kind="clone", src=b, family=b,
                             target_iou=round(tgt[b], 4), **info))
            # ---- exact pre-audit: this clone against every tile known so far
            if pa.add(m2, b):
                why["_pre"] = pa.reason
                return None
        # ---- a letter family is final now: its polarity gate (C >= 2) is decided
        pb = polarity_bad(dp, meta, families=[b])
        if pb:
            why["_pre"] = pb
            return None
    return masks, meta


def prescreen_ok(pre, tier, names):
    """The law's pre-screen (law/prescreen.json, `pre`): the fitted probability that an attempt drawing this
    identity set passes every gate must reach `tau`, else the set is redrawn before anything is built (it
    would almost surely be built for nothing).  The table is the trial fit refitted (offset method) with
    terms for the classes the per-node census rescued, which pass at about half the base rate.  tau is per
    rung where the table says so (0.05 on hard and xhard: -7% / -12% CPU per item, 1.1% / 1.5% of the
    accepted mass; 0.03 on xxhard, whose space is the tightest)."""
    if pre is None or tier not in pre["tiers"]:
        return True
    t = pre["tiers"][tier]
    z = t["intercept"]
    for c in sorted(names):
        z += t["coef"].get(c, 0.0)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) >= t.get("tau", pre["tau"])
