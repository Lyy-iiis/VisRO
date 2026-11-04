"""Cache stage of find_the_shadow: the orientation-augmented natural-rival law (VISRO_FTS_CACHE/law/).

WHY
---
The natural-rival rungs (hard / xhard / xxhard) key an item by its set of base
silhouettes, drawn as disjoint natural-rival edges of a graph.  Over the upright
icons alone that graph (59 / 42 / 36 edges over 77 / 60 / 51 icons) has fewer
possible keys than a 25,000-item cell needs.  This stage enlarges the graph
along the orientation of the base icons (what a reader sees changes; no
difficulty knob moves) and writes the selection tables the adapter draws from.

STAGES (each skipped when its output exists)
  orient  orient_92.npz        every drawable silhouette row (fill >= 0.36 at 92 px,
                               style silhouette: 1267 rows) rendered ONCE at 504 px
                               through the delivery path (canon_tile(render_canon(
                               ..., 504, black), 504), alpha > 40, as build_cache's
                               stage rok), each of the 8 orientations of the square
                               applied to that mask as an exact pixel permutation,
                               then norm_mask 160 -> 92 (stage retab); stored
                               sparsely: every node pair with M >= 0.62 or OUT >= 0.80
                               (all other pairs are ALLOWed, not BAND and not SB in
                               puzzle.select_tables) with M, OUT and RAD.  The
                               orientation-0 block equals Mr/OUTr/RADr_92 (asserted).
  census  strikes_law.json   build_strikes.py's census (same rule: K = 4 trials of
                               the family built on the node alone, strike on the first
                               failure) PER NODE (icon, orientation): upright nodes use
                               strikes_92.json where it has the row (rng "fts-strike/
                               <path>/<key>/<trial>") and the same seeds otherwise; a
                               node in orientation g > 0 uses rng "fts-strike/<path>@<g>/
                               <key>/<trial>" and is judged by its OWN trials only
                               (synthesis is not orientation-invariant either way:
                               file-download passes upright and fails in 6 of 7 other
                               orientations, and many icons struck upright pass rotated or
                               mirrored).  Candidates: every node of an edge of the
                               PRISTINE augmented graph.  Incremental: entries of an
                               existing strikes_law.json are reused (each trial is seeded
                               by its node, key and trial, so a job's result never depends
                               on the run that computed it).
  law     law/law.json, law/sel_{hard,xhard,xxhard}.npz
                               nodes, identity classes, conflicts and per-tier
                               selection tables (see law_sampler.py).
  verify  law/find_the_shadow_verify_table.json
                               the verifier's identity-class table (stage_verify_table).

ORIENTATION CODES (numpy on a mask m / PIL op on the square tile)
  0 m                 -            1 m[:, ::-1]   FLIP_LEFT_RIGHT   2 m[::-1]  FLIP_TOP_BOTTOM
  3 m[::-1, ::-1]     ROTATE_180   4 rot90(m, 1)  ROTATE_90         5 rot90(m, -1) ROTATE_270
  6 m.T               TRANSPOSE    7 m[::-1, ::-1].T  TRANSVERSE

    VISRO_FTS_CACHE=<cache> VISRO_FTS_WORKERS=6 python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_law [orient census law verify]
"""
import os, sys, re, json, time, random, hashlib, collections

import numpy as np
import cv2

import shared.iconlib as iconlib

from .. import decoys, icon_pool, puzzle
from ..base import silhouette
from ..cache import CACHE_DIR

if not iconlib.HAVE_CAIROSVG:
    raise SystemExit("build_law.py needs cairosvg")

NPROC = int(os.environ.get("VISRO_FTS_WORKERS", 6))
OUT_DIR = CACHE_DIR / "law"
GROUP = tuple(range(8))         # all 8 orientations of the square
TWIN_DISC = 1.5                 # in-orbit duplicate of a symmetric icon (the adapter's twin rule)
KEY_DISC = 2.5                  # identity classes: twins in some orientation within 2.5 px (cards merge)
CONFLICT_DISC = 3.5             # verifier: "shadow is pattern rotated or mirrored" at <= 3 px
KEYS = {"hard2": ("hard", 2), "xhard3": ("xhard", 3), "xxhard4": ("xxhard", 4), "xxhard3": ("xxhard", 3)}
TIER_KEYS = {"hard": ["hard2"], "xhard": ["xhard3"], "xxhard": ["xxhard4", "xxhard3"]}
LAW_TIERS = ("hard", "xhard", "xxhard")
# TIER_DROP: identity classes left out of one rung's graph to keep the rung at its difficulty level
# (measured on >= 398 fresh builds per rung with a solver-based scorer).  Every class listed is one the
# orientation pool brings into that rung (the per-node census rescues it, or a rotation makes it a natural
# rival there):
#   xhard   the two trophy rival pairs (rescued): items holding one score +0.13 on the area shortcut (area_best)
#           and -0.03 / -0.04 on warp2 / warp4.
#   xxhard  the rescued classes that read harder -- bookmark-plus / panorama (a rival pair) and navigation
#           (-0.03 warp4), the trophy-variant pair (-0.02 warp4, +0.06 area_best), soul-vessel (-0.02 warp2, a hub
#           in 20% of items) -- and the account-circle / circle-half-full pair, a disc rotated against a disc
#           (-0.03 warp4 and warp2 per item, in 17% of items).
TIER_DROP = {"xhard": ("mdi/svg/trophy-outline.svg", "mdi/svg/trophy-variant-outline.svg",
                       "mdi/svg/trophy-variant.svg", "mdi/svg/trophy.svg"),
             "xxhard": ("gameicons/delapouite/soul-vessel.svg", "mdi/svg/account-circle.svg",
                        "mdi/svg/bookmark-plus.svg", "mdi/svg/circle-half-full.svg", "mdi/svg/navigation.svg",
                        "mdi/svg/panorama.svg", "mdi/svg/trophy-variant-outline.svg", "mdi/svg/trophy-variant.svg")}
# text glyphs (letters, digits, brand marks with letters, playing cards, zodiac signs, ...): upright only
GLYPH_RE = re.compile(
    r"(^|-)(alpha|numeric|roman|sigma|omega|abugida|syllabary|format|currency|size|temperature|language|at|"
    r"comma|floor|24|text|excel|word|onenote|sharepoint|kilogram|gram|question|information|facebook|twitter|"
    r"google|gmail|quora|skype|netflix|hulu|vimeo|pinterest|deviantart|drupal|opera|gatsby|bulma|salesforce|"
    r"ubisoft|mastodon|dolby|emby|spotify|unity|trello|webrtc|digital|nativescript|svg|powershell|subtitles|"
    r"vhs|hspa|dlna|32|bit|zodiac|virgo|xamarin|humble|percent|parking|hospital|ev|transcribe|braces|card|"
    r"letter|kelvin|code|null|sale|xbox|youtube|android|apple|linux|git|css3|html5|rust|swift|fortran|"
    r"tally|khanda|new|id|sim|store|level|hiragana|thai|equal|accidental|sharp|music|"
    r"gemini|pisces|sagittarius|scorpio|taurus|ophiuchus|alert)($|-)")
# ("alert": mdi alert-* icons carry a "!" -- upside down or sideways it reads as a reversed character)


def orient(m, g):
    return [m, m[:, ::-1], m[::-1, :], m[::-1, ::-1], np.rot90(m, 1), np.rot90(m, -1), m.T,
            m[::-1, ::-1].T][g]


_S = {}


def load():
    """The release tables exactly as the adapter loads them (upright rows)."""
    if "tab" in _S:
        return _S["tab"]
    ents = icon_pool.load()[0]
    keep = np.load(CACHE_DIR / "keep3.npy"); sub = np.load(CACHE_DIR / "sub.npy")
    icons = [int(keep[i]) for i in sub]
    sil = np.array([ents[i]["style"] == "silhouette" for i in icons])
    M = np.load(CACHE_DIR / "Mr_92.npy"); RAD = np.load(CACHE_DIR / "RADr_92.npy")
    OUT = np.load(CACHE_DIR / "OUTr_92.npy"); fill = np.load(CACHE_DIR / "fillr_92.npy")
    from shared.iconlib import IconLibrary
    tab = dict(ents=ents, icons=icons, sil=sil, M=M, RAD=RAD, OUT=OUT, fill=fill,
               lib=IconLibrary(default_split="train"))
    tab["rows"] = np.flatnonzero((fill >= puzzle.BASE["fill_min"]) & sil)
    tab["paths"] = [ents[icons[int(r)]]["path"] for r in tab["rows"]]
    _S["tab"] = tab
    return tab


# ------------------------------------------------------------------ 1. orient
def _render(r):
    tab = _S["tab"]
    e = tab["ents"][tab["icons"][int(r)]]
    img = puzzle.canon_tile(silhouette.render_canon(tab["lib"], e, 504, (0, 0, 0)), 504)
    m = np.asarray(img)[..., 3] > 40
    return int(r), [np.packbits(decoys.norm_mask(np.ascontiguousarray(orient(m, g)), 160).ravel())
                    for g in range(8)]


def _block(lo_hi):
    lo, hi = lo_hi
    F, OF, NM = _S["F"], _S["OF"], _S["NM"]
    area, oa = F.sum(1), OF.sum(1)
    inter = F[lo:hi] @ F.T
    M = (inter / np.maximum(area[lo:hi, None] + area[None, :] - inter, 1.0)).astype(np.float32)
    oi = OF[lo:hi] @ OF.T
    O = (oi / np.maximum(oa[lo:hi, None] + oa[None, :] - oi, 1.0)).astype(np.float32)
    ii, jj = np.nonzero((M >= 0.62) | (O >= 0.80))
    ii = ii + lo
    keep = jj > ii
    ii, jj = ii[keep], jj[keep]
    mv, ov = M[ii - lo, jj], O[ii - lo, jj]
    rv = np.zeros(len(ii), np.float32)
    for k in np.flatnonzero(mv >= 0.62):
        x = np.logical_xor(NM[ii[k]], NM[jj[k]]).astype(np.uint8)
        rv[k] = cv2.distanceTransform(x, cv2.DIST_L2, 5).max()
    return ii.astype(np.int32), jj.astype(np.int32), mv, ov, rv


def stage_orient():
    out = CACHE_DIR / "orient_92.npz"
    if out.exists():
        print("orient_92.npz exists, skip"); return
    from multiprocessing import get_context
    tab = load(); rows = tab["rows"]; n = len(rows)
    t0 = time.time(); got = {}
    with get_context("fork").Pool(NPROC) as p:
        for r, packs in p.imap_unordered(_render, rows, chunksize=4):
            got[r] = packs
    m160 = np.stack([np.unpackbits(got[int(r)][g])[:160 * 160].reshape(160, 160).astype(bool)
                     for g in range(8) for r in rows])
    NM = np.stack([decoys.norm_mask(m, 92) for m in m160])
    _S["NM"] = NM
    _S["F"] = NM.reshape(len(NM), -1).astype(np.float32)
    _S["OF"] = np.stack([decoys.outline_fill(m) for m in NM]).reshape(len(NM), -1).astype(np.float32)
    N = len(NM)
    with get_context("fork").Pool(NPROC) as p:
        parts = p.map(_block, [(lo, min(N, lo + 500)) for lo in range(0, N, 500)])
    I, J, MV, OV, RV = (np.concatenate([x[k] for x in parts]) for k in range(5))
    sel = (I < n) & (J < n)
    sub = np.ix_(rows, rows)
    assert np.array_equal(MV[sel], tab["M"][rows[I[sel]], rows[J[sel]]])
    assert np.array_equal(OV[sel], tab["OUT"][rows[I[sel]], rows[J[sel]]])
    r62 = MV[sel] >= 0.62
    assert np.array_equal(RV[sel][r62], tab["RAD"][rows[I[sel]], rows[J[sel]]][r62])
    assert int(sel.sum()) == int(np.triu((tab["M"][sub] >= 0.62) | (tab["OUT"][sub] >= 0.80), 1).sum())
    np.savez(out, rows=rows, n=n, I=I, J=J, M=MV, OUT=OV, RAD=RV)
    print(f"orient: {N} nodes, {len(I)} pairs, orientation-0 block == shipped tables, {time.time() - t0:.0f}s",
          flush=True)


# ------------------------------------------------------------------ pool
class Pool:
    """Nodes, identity classes, conflicts and per-tier selection tables of the law."""

    def __init__(self, struck=None):
        tab = load()
        Z = np.load(CACHE_DIR / "orient_92.npz")
        self.rows, self.n = Z["rows"], int(Z["n"])
        n = self.n; N = 8 * n
        I, J, Mv, Ov, Rv = Z["I"], Z["J"], Z["M"], Z["OUT"], Z["RAD"]
        self.I, self.J, self.Mv, self.Ov, self.Rv = I, J, Mv, Ov, Rv
        self.paths = tab["paths"]
        self.icon_of = np.tile(np.arange(n), 8); self.g_of = np.repeat(np.arange(8), n)
        icon_of, g_of = self.icon_of, self.g_of
        names = [p.rsplit("/", 1)[-1].replace(".svg", "") for p in self.paths]
        self.glyph = np.array([bool(GLYPH_RE.search(x)) for x in names])
        gset = np.isin(g_of, GROUP) & ~(self.glyph[icon_of] & (g_of != 0))
        same_icon = icon_of[I] == icon_of[J]
        tw = (Mv >= 0.9) & (Rv <= TWIN_DISC)
        dup = np.zeros(N, bool)
        for a, b in zip(I[tw & same_icon], J[tw & same_icon]):
            lo, hi = (a, b) if g_of[a] < g_of[b] else (b, a)
            if gset[lo] and gset[hi]:
                dup[hi] = True
        self.node_ok = gset & ~dup
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        kt = (Mv >= 0.9) & (Rv <= KEY_DISC) & ~same_icon
        for a, b in zip(I[kt], J[kt]):
            par[find(int(icon_of[a]))] = find(int(icon_of[b]))
        mem = collections.defaultdict(list)
        for i in range(n):
            mem[find(i)].append(i)
        self.ident = np.zeros(n, np.int64); self.ident_name = [None] * n
        for ms in mem.values():
            nm = min(self.paths[i] for i in ms)
            for i in ms:
                self.ident[i] = min(ms); self.ident_name[i] = nm
        near = (Mv >= 0.62) & (Rv <= CONFLICT_DISC) & ~same_icon
        self.conf = collections.defaultdict(set)
        for a, b in zip(I[near], J[near]):
            i, ga = int(icon_of[a]), int(g_of[a]); j, gb = int(icon_of[b]), int(g_of[b])
            for g in range(8):
                if g != ga:
                    self.conf[g * n + i].add(int(b)); self.conf[int(b)].add(g * n + i)
                if g != gb:
                    self.conf[g * n + j].add(int(a)); self.conf[int(a)].add(g * n + j)
        self.struck = struck

    def tier_tables(self, tier, pristine=False):
        dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
        thr = dp["rad_pre"]
        avail = self.node_ok.copy()
        if not pristine:
            avail &= ~self.struck[tier]
        I, J, Mv, Ov, Rv = self.I, self.J, self.Mv, self.Ov, self.Rv
        idn = self.ident[self.icon_of]
        fail = ((Mv >= 0.62) & (Rv < thr)) | (Mv > dp["hi_nat"]) | ((Mv >= 0.80) & (Ov > dp["out_max"]))
        band = ~fail & (Mv >= dp["mult_thr"]) & avail[I] & avail[J] & (idn[I] != idn[J])
        bi, bj = I[band], J[band]
        okc = np.array([int(b) not in self.conf.get(int(a), ()) for a, b in zip(bi, bj)], bool)
        bi, bj = bi[okc], bj[okc]
        nodes = np.unique(np.concatenate([bi, bj]))
        pos = {int(x): k for k, x in enumerate(nodes)}
        R = len(nodes)
        ALLOW = np.ones((R, R), bool); SB = np.zeros((R, R), bool)
        inR = np.isin(I, nodes) & np.isin(J, nodes)
        for a, b, f, o in zip(I[inR], J[inR], fail[inR], Ov[inR]):
            pa, pb = pos[int(a)], pos[int(b)]
            if f:
                ALLOW[pa, pb] = ALLOW[pb, pa] = False
            if o >= 0.93:
                SB[pa, pb] = SB[pb, pa] = True
        idR = idn[nodes]
        ALLOW &= idR[:, None] != idR[None, :]
        for k, x in enumerate(nodes):
            for y in self.conf.get(int(x), ()):
                if y in pos:
                    ALLOW[k, pos[y]] = ALLOW[pos[y], k] = False
        np.fill_diagonal(ALLOW, False); np.fill_diagonal(SB, False)
        BAND = np.zeros((R, R), bool)
        BAND[[pos[int(a)] for a in bi], [pos[int(b)] for b in bj]] = True
        BAND |= BAND.T
        BAND &= ALLOW
        E = np.argwhere(np.triu(BAND, 1))
        return dp, nodes, ALLOW, SB, BAND, E


# ------------------------------------------------------------------ 2. census
def _census_one(job):
    key, row, g = job
    tab = _S["tab"]
    tier, c = KEYS[key]
    dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
    dp.update(L=1, NU=0, letter_clones=[c], u_clones=[])
    path = tab["ents"][tab["icons"][row]]["path"]
    tag = path if g == 0 else f"{path}@{g}"
    fails = 0
    for trial in range(4):
        rng = random.Random(f"fts-strike/{tag}/{key}/{trial}")
        if puzzle.build_shadows(tab["lib"], tab["ents"], [(tab["icons"][row], g)], dp, rng) is None:
            fails += 1
            break
    return key, row, g, [fails, trial + 1]


def stage_census():
    out = CACHE_DIR / "strikes_law.json"
    from multiprocessing import get_context
    tab = load()
    shipped = json.loads((CACHE_DIR / "strikes_92.json").read_text())["census"]
    have = json.loads(out.read_text())["census"] if out.exists() else {}
    pool = Pool()
    jobs = []
    for key, (tier, c) in KEYS.items():
        dp, nodes, *_ = pool.tier_tables(tier, pristine=True)
        rows = sorted({int(pool.rows[pool.icon_of[x]]) for x in nodes})
        jobs += [(key, r, 0) for r in rows if str(r) not in shipped[key]]
        jobs += [(key, int(pool.rows[pool.icon_of[x]]), int(pool.g_of[x])) for x in nodes if pool.g_of[x] != 0]
    res = {k: {} for k in KEYS}
    todo = []
    for key, row, g in jobs:
        tag = str(row) if g == 0 else f"{row}@{g}"
        if tag in have.get(key, {}):
            res[key][tag] = have[key][tag]
        else:
            todo.append((key, row, g))
    print(f"census: {len(jobs)} node jobs, {len(jobs) - len(todo)} reused, {len(todo)} to run", flush=True)
    with get_context("fork").Pool(NPROC) as p:
        for key, row, g, v in p.imap_unordered(_census_one, todo, chunksize=1):
            res[key][str(row) if g == 0 else f"{row}@{g}"] = v
    doc = dict(format="fts-strikes@3", K=4,
               rule=("strike a node when fails >= 1 for any key of its rung; every node '<row>' / '<row>@<g>' is "
                     "judged by its own trials (upright rows: strikes_92.json where it has them)"),
               scope="nodes (row or row@orientation) of the orientation-augmented graph not in strikes_92.json",
               n_rows=len(tab["icons"]), tier_keys=TIER_KEYS,
               census={k: dict(sorted(v.items())) for k, v in res.items()})
    out.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    print("census:", {k: f"{sum(1 for v in d.values() if v[0])}/{len(d)} struck" for k, d in res.items()}, flush=True)


# ------------------------------------------------------------------ 3. law
def stage_law():
    if (OUT_DIR / "law.json").exists():
        print("law/law.json exists, skip"); return
    tab = load()
    census = {}
    for f in ("strikes_92.json", "strikes_law.json"):
        for k, d in json.loads((CACHE_DIR / f).read_text())["census"].items():
            census.setdefault(k, {}).update(d)
    if json.loads((CACHE_DIR / "strikes_law.json").read_text()).get("format") != "fts-strikes@3":
        raise SystemExit("strikes_law.json is not fts-strikes@3 (per-node census); rerun the census stage")
    pos = {int(r): i for i, r in enumerate(tab["rows"])}
    n = len(tab["rows"])
    struck = {}
    for t, keys in TIER_KEYS.items():
        s = np.zeros(8 * n, bool)                  # per node g * n + i
        for k in keys:
            for r, (fails, _tr) in census[k].items():
                if not fails:
                    continue
                row, _, g = r.partition("@")
                if int(row) not in pos:
                    continue
                i = pos[int(row)]
                s[int(g or 0) * n + i] = True      # the node itself; its other orientations have their own trials
        struck[t] = s
    pool = Pool(struck)
    for t, drop in TIER_DROP.items():
        if not set(drop) <= set(pool.ident_name):
            raise SystemExit(f"TIER_DROP[{t}] names a class this cache does not have")
        cut = np.isin(np.array(pool.ident_name, dtype=object), list(drop))
        struck[t][np.tile(cut, 8)] = True          # every node of every icon of a dropped identity class
    OUT_DIR.mkdir(exist_ok=True)
    doc = dict(format="fts-law@1", group=list(GROUP), twin_disc=TWIN_DISC, key_disc=KEY_DISC,
               conflict_disc=CONFLICT_DISC, n=pool.n, rows=[int(r) for r in pool.rows], paths=pool.paths,
               ident=list(pool.ident_name), glyph=[bool(x) for x in pool.glyph],
               tier_drop={t: list(v) for t, v in sorted(TIER_DROP.items())},
               tiers={})
    for tier in LAW_TIERS:
        dp, nodes, ALLOW, SB, BAND, E = pool.tier_tables(tier)
        p = OUT_DIR / f"sel_{tier}.npz"
        cls = pool.ident[pool.icon_of[nodes]].astype(np.int32)
        np.savez(p, nodes=nodes.astype(np.int32), cls=cls, ALLOW=np.packbits(ALLOW, axis=1),
                 SB=np.packbits(SB, axis=1), BAND=np.packbits(BAND, axis=1), E=E.astype(np.int32),
                 R=np.int32(len(nodes)))
        doc["tiers"][tier] = dict(nodes=int(len(nodes)), edges=int(len(E)),
                                  icons=len({int(pool.icon_of[x]) for x in nodes}),
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (OUT_DIR / "law.json").write_text(json.dumps(doc))
    print("law:", json.dumps(doc["tiers"]), flush=True)


# ------------------------------------------------------------------ 4. the verifier's table
def stage_verify_table():
    """the find_the_shadow law table from law/law.json: the identity classes
    (representative = smallest path: the other members) and the text glyphs drawn upright only, over the
    1267 drawable rows.  Written to OUT_DIR/find_the_shadow_verify_table.json; copy it next to the verifier
    and pin its sha256 there (VERIFY_TABLE_SHA256)."""
    doc = json.loads((OUT_DIR / "law.json").read_text())
    mem = collections.defaultdict(list)
    for p, c in zip(doc["paths"], doc["ident"]):
        mem[c].append(p)
    classes = {c: sorted(x for x in ms if x != c) for c, ms in sorted(mem.items()) if len(ms) > 1}
    glyph = sorted(p for p, g in zip(doc["paths"], doc["glyph"]) if g)
    out = dict(classes=classes, format="fts-law-verify@1", glyph_upright_only=glyph,
               key_disc_px=doc["key_disc"],
               note=("identity classes (representative = smallest path: other members) and the text-glyph icons "
                     "drawn upright only, for the natural-rival tiers of find_the_shadow; derived from the "
                     "cache's law/law.json by build_law.py"),
               orientations=doc["group"])
    p = OUT_DIR / "find_the_shadow_verify_table.json"
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("verify table:", p, hashlib.sha256(p.read_bytes()).hexdigest(), flush=True)


STAGES = dict(orient=stage_orient, census=stage_census, law=stage_law, verify=stage_verify_table)

if __name__ == "__main__":
    cv2.setNumThreads(1)
    for s in sys.argv[1:] or list(STAGES):
        print("== stage", s, "==", flush=True)
        STAGES[s]()
