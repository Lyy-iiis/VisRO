"""find_the_shadow -- match each coloured pattern to its shadow among look-alikes.

Generator: generator/puzzle.py (the five-rung ladder, (index, orientation) icon
refs) with decoys.py (clone synthesis), icon_pool.py (the cached icon pool),
law_sampler.py (the sampler of the natural-rival rungs) and speedups.py
(output-preserving fast paths).  The TIERS table, the ACCEPT floors,
select_base / select_base_free, build_shadows, render and audit are called as
they are; `_candidate` below runs the per-seed gate stack.

    * One item is a pure function of (tier, seed): the search walks sub-seeds
      sha256(seed, attempt).  Two items never share a random stream, whatever
      the tier, split or index.
    * Icons whose look-alike family fails to synthesise are left out of the
      rung's pool by a table precomputed once per cache
      (generator/build/build_strikes.py: every icon a rung can draw is given K = 4
      fixed-seed tries at building its family with that rung's parameters, and
      an icon that fails any of them is not drawn on that rung).  It removes
      icons, never gates.  Duplicates are handled by the content core_key (see
      build()).
    * Every icon is rasterised by cairosvg; no cairosvg -> the adapter refuses
      to run.

THE CACHE (not in git; about 154 MB as built, of which generation reads the
139 MB in CACHE_REQUIRED, plus orient_92.npz, strikes_law.json and law/ --
see generator/law_sampler.py and generator/build/build_law.py)
    The tables are built by generator/build/build_cache.py (about 20 min on 6
    processes, most of it the strike census); the adapter reads the 7 of them
    in CACHE_REQUIRED, plus strikes_92.json from build_strikes.py (build_cache.py
    runs it as its last stage).  docs/ASSETS.md gives the commands.  The
    adapter looks for them in
        $VISRO_FTS_CACHE, else ~/.cache/visro/find_the_shadow
    and raises FileNotFoundError (never a silent rebuild) when a table is
    missing.  The .npy tables are fingerprinted; a cache that does not match
    the one this release was generated with raises too, unless
    VISRO_FTS_ALLOW_CACHE_DRIFT=1.  The release pool is build_cache.py's,
    upright icons of style "silhouette".
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from visro import api
from visro.seeding import digest
from tasks.discrimination_grouping.find_the_shadow.generator.cache import BUILD_HINT, CACHE_DIR

TASK = "find_the_shadow"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

# Pool sizes are what select_base / select_base_free can actually draw from on
# each rung: fill >= 0.36 silhouettes of the cached train-icon pool (1267) minus
# the rung's strike census.  On the natural=on rungs every base icon is an
# endpoint of a natural-rival edge with BOTH endpoints drawable; the counts are
# those edges and icons (before the strike census: 103 edges over 128 icons).
_SPLITS_NOTE = ("; every split draws from iconlib's train icons: val/eval/ood are disjoint from train by "
                "base-icon set (core_key) and image only, not by icon")
TIER_DEFINITIONS = {
    "easy": ("puzzle TIERS['easy']: 644x364 px, 3x6 cells, 92 px tiles; L=5 patterns A-E; "
             "N=13 shadows = 5 true + 5 clones (C=1 per letter) + answerless family (base + 1 clone) "
             "+ 1 lone, U=8 unmatched; natural=off: bases drawn freely with pairwise IoU<0.62 from 910 "
             "drawable silhouettes (fill>=0.36, not struck by the c1 census)"),
    "medium": ("puzzle TIERS['medium']: 728x364 px, 3x7 cells, 92 px tiles; L=6 patterns A-F; "
               "N=15 shadows = 6 true + 6 clones (C=1) + answerless family (base + 1 clone) + 1 lone, "
               "U=9; natural=off (pairwise base IoU<0.62), same 910 drawable silhouettes as easy"),
    "hard": ("puzzle TIERS['hard']: 728x476 px, 4x7 cells, 92 px tiles; L=6 patterns A-F; "
             "N=22 shadows = 6 true + 12 clones (C=2) + answerless family (base + 2 clones) + 1 lone, "
             "U=16; natural=on: bases are 4 disjoint natural-rival pairs (gate IoU 0.72-0.855) drawn "
             "class-balanced from the 852 rival edges over 125 icons (483 icon orientations) of the "
             "orientation-augmented pool left drawable by the per-orientation hard2 strike census"),
    "xhard": ("puzzle TIERS['xhard']: 728x616 px, 5x7 cells, 92 px tiles; L=6 patterns A-F; "
              "N=29 shadows = 6 true + 18 clones (C=3) + answerless family (base + 3 clones) + 1 lone, "
              "U=23; natural=on: 4 disjoint natural-rival pairs drawn class-balanced from the 695 rival "
              "edges over 103 icons (420 orientations) left drawable by the per-orientation xhard3 strike census, "
              "the two trophy rival pairs left out"),
    "xxhard": ("puzzle TIERS['xxhard']: 728x616 px, 5x7 cells, 92 px tiles; L=5 patterns A-E; "
               "N=30 shadows = 5 true + 20 clones (C=4) + answerless family (base + 3 clones) + 1 lone, "
               "U=25; natural=on: 3 disjoint natural-rival pairs + 1 rival of them drawn class-balanced "
               "from the 561 rival edges over 90 icons (356 orientations) left drawable by the per-orientation "
               "xxhard4/xxhard3 strike census, 8 identity classes left out"),
}
# Shared by all five rungs (puzzle BASE / ACCEPT / R_TARGET): every clone is
# one rounded notch or lobe of designed radius 12 px on the outer edge, clone
# IoU in [0.86, 0.945], every wrong candidate refuted by an inscribed disc
# >= 8 px in puzzle.audit (bbox-normalised masks of the rendered PNG, disc
# over the unpadded 92 px tile), true pair differs by <= 1.0 px, polarity mixed
# inside every family (C>=2) and across the letter clones, area_rule_hits < L.
for _t in TIER_DEFINITIONS:
    TIER_DEFINITIONS[_t] += ("; shared floors: clone IoU 0.86-0.945 with one 12 px designed edit on the outer "
                             "edge, every wrong candidate refuted by a >=8 px disc and true pair <=1 px in the "
                             "generator's audit" + _SPLITS_NOTE)

# Seed-search budget per item (attempts).  With the strike table an accepted
# item takes a few attempts on average (a few tens at most).  An index that
# exhausts its budget is a counted Reject, and 250 bounds a stalled worker to
# ~10 min.
BUDGET = {t: 250 for t in TIER_DEFINITIONS}

CACHE_REQUIRED = ("pool3.pkl", "keep3.npy", "sub.npy",
                  "Mr_92.npy", "RADr_92.npy", "OUTr_92.npy", "fillr_92.npy", "strikes_92.json")
# sha256 over the four gate tables + keep3 + sub + strikes_92.json (in that
# order), and over the ordered list of pool icon paths, as built by
# build_cache.py + build_strikes.py for this release.  The .npy tables come out
# identical under numpy 1.26 / cv2 4.10 / cairosvg 2.9.0 and numpy 2.5.4 /
# cv2 5.0 / cairosvg 2.9.1 (measured); strikes_92.json was built under the latter.
CACHE_FINGERPRINT = "b2bd08d618400be5cdb4eefc4bc9632c052a9ba94658c5cdf14a952c666ea10e"
POOL_FINGERPRINT = "22c9c064b105972f4dfb23a74aba0cc2630bf9cc81751ae1644e02970133f744"


def _missing() -> list[str]:
    return [f for f in CACHE_REQUIRED if not (CACHE_DIR / f).is_file()]


_TAB = None        # loaded tables
_T_CACHE = {}      # per-tier select_tables (pristine, never mutated)
icon_pool = law_sampler = puzzle = None     # the generator modules, imported by _import_gen()


def _import_gen():
    """Import the generator modules and install their output-preserving fast paths."""
    global icon_pool, law_sampler, puzzle
    if puzzle is not None:
        return
    miss = _missing()
    if miss:
        raise FileNotFoundError(
            f"find_the_shadow cache incomplete in {CACHE_DIR}: missing {miss}; " + BUILD_HINT)
    import shared.iconlib as iconlib
    if not iconlib.HAVE_CAIROSVG:
        raise RuntimeError("find_the_shadow needs cairosvg: the release renders every icon live")
    try:
        import cv2
        cv2.setNumThreads(1)          # 6 workers must mean 6 cores
    except Exception:  # noqa: BLE001
        pass
    from tasks.discrimination_grouping.find_the_shadow.generator import icon_pool, law_sampler, puzzle, speedups
    # caches and crop-local edits, no change to any byte
    speedups.install()


def _fingerprints(ents):
    h = hashlib.sha256()
    for f in ("Mr_92.npy", "RADr_92.npy", "OUTr_92.npy", "fillr_92.npy", "keep3.npy", "sub.npy",
              "strikes_92.json"):
        h.update((CACHE_DIR / f).read_bytes())
    p = hashlib.sha256(json.dumps([e["path"] for e in ents]).encode()).hexdigest()
    return h.hexdigest(), p


def _tables():
    global _TAB
    if _TAB is not None:
        return _TAB
    _import_gen()
    ents = icon_pool.load()[0]             # pool3.pkl exists (checked), so this only unpickles
    cfp, pfp = _fingerprints(ents)
    drift = []
    if CACHE_FINGERPRINT and cfp != CACHE_FINGERPRINT:
        drift.append(f"gate tables {cfp[:16]} != {CACHE_FINGERPRINT[:16]}")
    if POOL_FINGERPRINT and pfp != POOL_FINGERPRINT:
        drift.append(f"icon pool {pfp[:16]} != {POOL_FINGERPRINT[:16]}")
    if drift and os.environ.get("VISRO_FTS_ALLOW_CACHE_DRIFT") != "1":
        raise RuntimeError(f"find_the_shadow cache in {CACHE_DIR} is not the release cache "
                           f"({'; '.join(drift)}); rebuild it or set VISRO_FTS_ALLOW_CACHE_DRIFT=1")
    keep = np.load(CACHE_DIR / "keep3.npy")
    sub = np.load(CACHE_DIR / "sub.npy")
    icons = [int(keep[i]) for i in sub]
    sil = np.array([ents[i]["style"] == "silhouette" for i in icons])
    M = np.load(CACHE_DIR / "Mr_92.npy")
    RAD = np.load(CACHE_DIR / "RADr_92.npy")
    OUT = np.load(CACHE_DIR / "OUTr_92.npy")
    fill = np.load(CACHE_DIR / "fillr_92.npy")
    strikes = json.loads((CACHE_DIR / "strikes_92.json").read_text())
    if strikes.get("format") != "fts-strikes@1" or strikes.get("n_rows") != len(icons):
        raise RuntimeError(f"{CACHE_DIR}/strikes_92.json does not belong to this cache; rerun build_strikes.py")
    struck = {}
    for tier, keys in strikes["tier_keys"].items():
        bad = np.zeros(len(icons), bool)
        for k in keys:
            for row, (fails, _trials) in strikes["census"][k].items():
                if fails:
                    bad[int(row)] = True
        struck[tier] = bad
    twin_rep = _twin_classes(M, RAD, fill, sil, ents, icons)
    tsha = _twin_table_sha256(twin_rep)
    if tsha != TWIN_TABLE_SHA256 and os.environ.get("VISRO_FTS_ALLOW_CACHE_DRIFT") != "1":
        raise RuntimeError(f"find_the_shadow silhouette-class table {tsha[:16]} != {TWIN_TABLE_SHA256[:16]} "
                           f"(the verifier's SILHOUETTE_CLASSES); its records would not verify")
    from shared.iconlib import IconLibrary
    # The cached pool (build/build_cache.py) holds train icons only, and every split
    # draws from it: `split` reaches build() only through the derived seed.
    # iconlib's held-out test icons (1163 silhouettes) were never cached, so
    # no split holds out icons (see the task README).
    lib = IconLibrary(default_split="train")
    _TAB = dict(ents=ents, icons=icons, sil=sil, M=M, RAD=RAD, OUT=OUT, fill=fill, lib=lib,
                struck=struck, twin_rep=twin_rep, cache_fp=cfp, pool_fp=pfp)
    return _TAB


# Two icon files can draw the same silhouette: mdi rhombus / rhombus-medium and
# truck-plus / ambulance at IoU 1.0, playing cards of two suits, egg / big-egg,
# ...  ALLOW (IoU <= 0.855 unless the XOR disc is >= 11 px) keeps such twins
# out of one item, but a key built from file paths would let two items whose
# icon sets differ only by a twin both through.  The key is therefore built from
# silhouette classes: icons whose gate tables put them at
# IoU >= 0.9 with an inscribed XOR disc <= 1.5 px (antialiasing, the task's own
# "departs nowhere") are one class.  27 such pairs among the 1267 drawable
# icons give 41 icons in 18 classes.  The verifier declares the same table
# (SILHOUETTE_CLASSES) and requires every record's classes to be its classes;
# TWIN_TABLE_SHA256 pins it here, so a cache or rule change that moved a class
# fails loudly in _tables() instead of producing records the verifier rejects.
TWIN_IOU, TWIN_DISC = 0.9, 1.5
TWIN_TABLE_SHA256 = "79814584d43bd918c7e280267afd74ccd6edd8bc9c2c466aa9734c7f4acdd2fe"


def _twin_table_sha256(rep):
    s = json.dumps(dict(sorted(rep.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def _twin_classes(M, RAD, fill, sil, ents, icons):
    """{path: class representative} for every drawable icon that has a twin; the
    representative is the smallest path of its class (union-find over twin pairs)."""
    _import_gen()
    idx = np.flatnonzero((fill >= puzzle.BASE["fill_min"]) & sil)
    hit = (M[np.ix_(idx, idx)] >= TWIN_IOU) & (RAD[np.ix_(idx, idx)] <= TWIN_DISC)
    np.fill_diagonal(hit, False)
    parent = {int(r): int(r) for r in idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(*np.nonzero(np.triu(hit, 1))):
        parent[find(int(idx[i]))] = find(int(idx[j]))
    members = collections.defaultdict(list)
    for r in parent:
        members[find(r)].append(ents[icons[r]]["path"])
    rep = {}
    for paths in members.values():
        if len(paths) > 1:
            for path in paths:
                rep[path] = min(paths)
    return rep


# ---------------------------------------------------------------- law
# The natural-rival rungs draw their base icons from the law tables
# (generator/law_sampler.py), precomputed by generator/build/build_law.py into
# CACHE_DIR/law/{law.json, sel_<tier>.npz}; VISRO_FTS_LAW overrides the
# directory.
LAW_TIERS = ("hard", "xhard", "xxhard")
LAW_SUBDIR = "law"
LAW_FINGERPRINT = "b06d3e850b3a6a2727cf39d60bf636a3dca8c72869a892681766c78d67993a9f"   # law.json + sel_*.npz + prescreen.json
MAX_REDRAW = 4000               # redraws of an excluded identity set inside one attempt


_LAW = None
_EXCL = frozenset()             # cell-start snapshot (prebuild_exclusion)
_EXCL_INFO = dict(n_start=0, sha256_start=hashlib.sha256(b"").hexdigest())
_SEARCH = collections.Counter()  # per build(): redraws by cause (meta.search)
_HAS_ND = [False]               # does the exclusion hold near-duplicate keys (ND_PREFIX)?


def prebuild_exclusion(keys):
    """Keys already taken -- identity keys (core_key digests) of earlier cells, and
    the near-duplicate keys (ND_PREFIX, see neardup_keys) of eval/ood items: an attempt that draws a taken or
    near-duplicate identity set redraws it before anything is built, instead of costing a full build that the
    runner's dedup (or its near-duplicate rule) then rejects.  The runner hands this snapshot to the adapter at
    the start of a cell, before its workers fork; it replaces any earlier snapshot.  An item is then a pure
    function of (tier, seed, index, this set)."""
    global _EXCL, _EXCL_INFO
    ks = sorted({str(k) for k in keys if not str(k).startswith("png:")})
    _EXCL = frozenset(ks)
    _EXCL_INFO = dict(n_start=len(ks), sha256_start=hashlib.sha256("\n".join(ks).encode()).hexdigest())
    _HAS_ND[0] = any(k.startswith(ND_PREFIX) for k in ks)


def _taken(k):
    return k in _EXCL


# Near-duplicate keys (the cross-split rule).  An 8-icon candidate (train, or eval after eval) may not share 7 of
# its 8 identity classes with an eval item, nor contain all 7 classes of an ood item; a 7-icon (ood) candidate may
# not be contained in an eval item.  An item publishes keys (contributed): an eval 8-set the digests of its
# 7-subsets ("k7of8"), an ood 7-set the digest of itself ("k7of7"); a candidate 8-set tests the digests of its own
# 7-subsets under both tags, a candidate 7-set its own digest under "k7of8" (checked).  (A 6-of-7 rule would
# exclude ~19% of the train/xhard accepted mass.)  Used by the runner's side rule and, before building, by
# _candidate.
ND_PREFIX = "rule:fts-nd@3:"


def _nd_digest(tag, sub):
    return ND_PREFIX + tag + ":" + hashlib.sha256("\n".join(sorted(sub)).encode()).hexdigest()[:40]


def neardup_keys(names, role):
    """role "contributed": the keys an eval/ood item publishes; role "checked": the keys a candidate tests."""
    s = sorted(set(names))
    if len(s) not in (7, 8):
        return set()
    if role == "contributed":
        if len(s) == 8:
            return {_nd_digest("k7of8", tuple(x for x in s if x != y)) for y in s}
        return {_nd_digest("k7of7", tuple(s))}
    if len(s) == 8:
        sub7 = [tuple(x for x in s if x != y) for y in s]
        return {_nd_digest("k7of8", t) for t in sub7} | {_nd_digest("k7of7", t) for t in sub7}
    return {_nd_digest("k7of8", tuple(s))}          # a 7-set: not contained in an eval 8-set


def _law():
    global _LAW
    if _LAW is not None:
        return _LAW
    tab = _tables()
    d = Path(os.environ.get("VISRO_FTS_LAW") or (CACHE_DIR / LAW_SUBDIR))
    doc = json.loads((d / "law.json").read_text())
    h = hashlib.sha256((d / "law.json").read_bytes())
    tiers = {}
    for t in LAW_TIERS:
        raw = (d / f"sel_{t}.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != doc["tiers"][t]["sha256"]:
            raise RuntimeError(f"{d}/sel_{t}.npz does not match its law.json")
        h.update(raw)
        z = np.load(d / f"sel_{t}.npz")
        R = int(z["R"])
        unpack = lambda a: np.unpackbits(a, axis=1)[:, :R].astype(bool)
        tiers[t] = dict(nodes=z["nodes"], cls=z["cls"], ALLOW=unpack(z["ALLOW"]), SB=unpack(z["SB"]),
                        BAND=unpack(z["BAND"]), E=z["E"].astype(np.int64))
    pre = None
    if (d / "prescreen.json").is_file():
        raw = (d / "prescreen.json").read_bytes()
        h.update(raw)
        pre = json.loads(raw)
    fp = h.hexdigest()
    if LAW_FINGERPRINT and fp != LAW_FINGERPRINT and os.environ.get("VISRO_FTS_ALLOW_CACHE_DRIFT") != "1":
        raise RuntimeError(f"find_the_shadow law tables {fp[:16]} != {LAW_FINGERPRINT[:16]}")
    rows = doc["rows"]
    if [tab["ents"][tab["icons"][r]]["path"] for r in rows] != doc["paths"]:
        raise RuntimeError("law.json rows do not name this cache's icons")
    _LAW = dict(doc=doc, n=int(doc["n"]), rows=rows, ident=doc["ident"], tiers=tiers, fp=fp,
                prescreen=pre)
    return _LAW


def _law_key(names):
    return digest(dict(task=TASK, ident=sorted(names)))


def _tier_tables(tier):
    if tier not in _T_CACHE and tier in LAW_TIERS:
        _tables()
        law = _law()
        dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
        t = law["tiers"][tier]
        T = (t["ALLOW"], t["SB"], t["BAND"], t["E"], float(dp["rad_pre"]))
        _T_CACHE[tier] = (dp, np.ones(len(t["nodes"]), bool), T)
    if tier not in _T_CACHE:
        tab = _tables()
        dp = dict(puzzle.TIERS[tier]); dp["tier"] = tier
        # the rung's pool minus the icons build_strikes.py strikes
        # (build_strikes.py: the steady state of that eviction, as a table)
        avail = (tab["fill"] >= dp["fill_min"]) & tab["sil"] & ~tab["struck"][tier]
        T = puzzle.select_tables(tab["M"], tab["RAD"], tab["OUT"], avail, dp)
        _T_CACHE[tier] = (dp, avail, T)
    return _T_CACHE[tier]


def _sub_seed(seed: int, attempt: int) -> int:
    h = hashlib.sha256(f"find_the_shadow/{int(seed)}/{int(attempt)}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def _gates(dp, aud, meta):
    """The per-seed gate block of the find_the_shadow ladder."""
    L, C = dp["L"], dp["C"]
    A = puzzle.ACCEPT
    bad = []
    if not aud["unique_bijection"]: bad.append("bijection")
    if not aud["exhaustive_unique"]: bad.append("exhaustive")
    if min(aud["self_iou"]) < A["self_iou_min"]: bad.append("self_iou")
    if aud["min_disc_radius_px"] < A["disc_min"]: bad.append("disc")
    if aud["min_family_disc_px"] < A["disc_min"]: bad.append("clone_pair_disc")
    if max(aud["true_pair_disc_px"]) > A["true_disc_max"]: bad.append("true_flat")
    if min(aud["xor_ratio"]) < A["xor_ratio_min"]: bad.append("xor_ratio")
    if min(aud["outline_frac_rival"]) < A["outline_frac_min"]: bad.append("outline")
    if min(aud["rival_c_iou"]) < dp["clone_lo"]: bad.append("clone_band_lo")
    if max(aud["rival_iou"]) > dp["clone_hi"]: bad.append("clone_band_hi")
    for i in range(L):
        fam = {j for j, m in enumerate(meta) if m["kind"] == "clone" and m["src"] == i}
        if set(aud["topc_idx"][i]) != fam:
            bad.append("topc_not_family"); break
    pol = [m["polarity"] for m in meta if m["kind"] == "clone"]
    if min(pol.count("smoother"), pol.count("extra")) < 2:
        bad.append("polarity_mix")
    if C >= 2:
        for i in range(L):
            fp = [m["polarity"] for m in meta if m["kind"] == "clone" and m["src"] == i]
            if min(fp.count("smoother"), fp.count("extra")) < 1:
                bad.append("family_polarity"); break
    lp = [m["polarity"] for m in meta if m["kind"] == "clone" and m["src"] < L]
    if min(lp.count("smoother"), lp.count("extra")) < 1:
        bad.append("letter_polarity_mix")
    if aud["area_rule_hits"] >= L:
        bad.append("area_rule")
    if dp["natural"] == "off" and max(aud["rival_cplus1_iou"]) >= dp["free_pair_max"]:
        bad.append("free_rival")
    return bad


def _candidate(tier, s):
    """One seed of the search loop: a dict when every gate passes, else a reason."""
    tab = _tables()
    dp, avail, T = _tier_tables(tier)
    L, NU, NSHAD = dp["L"], dp["NU"], dp["NSHAD"]
    ents, icons, lib = tab["ents"], tab["icons"], tab["lib"]
    rng = random.Random(s)
    names = None
    if tier in LAW_TIERS:
        law = _law(); lt = law["tiers"][tier]; n = law["n"]
        nd = _HAS_ND[0]
        for _redraw in range(MAX_REDRAW):
            idxs = law_sampler.select_balanced(T, avail, L + NU, dp, rng, lt["cls"])
            if idxs is None:
                return "select"
            nodes = [int(lt["nodes"][i]) for i in idxs]
            names = [law["ident"][x % n] for x in nodes]
            # redraw (before anything is built) a set that is taken, a near-duplicate of an eval/ood item, or
            # below the pre-screen; the three tests only decide redraw-or-not, so their order changes nothing
            if _EXCL and _taken(_law_key(names)):
                _SEARCH["redraw_taken"] += 1
                continue
            if nd and any(_taken(k) for k in neardup_keys(names, "checked")):
                _SEARCH["redraw_neardup"] += 1
                continue
            if not law_sampler.prescreen_ok(law["prescreen"], tier, names):
                _SEARCH["redraw_prescreen"] += 1
                continue
            break
        else:
            return "excluded"
        refs = [(icons[law["rows"][x % n]], x // n) for x in nodes]
        why = {}
        pa = law_sampler.PreAudit(lib, ents, refs, dp)
        got = law_sampler.build_shadows_law(lib, ents, refs, dp, rng, why=why, pa=pa)
        if got is None:
            if why.get("_pre"):
                return "pre:" + why["_pre"]
            return "xdisc" if why.get("_xdisc") else "synth"
        pb = law_sampler.polarity_bad(dp, got[1])
        if pb:
            return "pre:" + pb
    else:
        pick = puzzle.select_base if dp["natural"] == "on" else puzzle.select_base_free
        idxs = pick(tab["M"], T, avail, L + NU, dp, rng)
        if idxs is None:
            return "select"
        refs = [(icons[i], False) for i in idxs]
        got = puzzle.build_shadows(lib, ents, refs, dp, rng)
        if got is None:
            return "synth"
    masks, smeta = got
    assert len(masks) == NSHAD
    rr = puzzle.render(lib, ents, refs, masks, dp, rng)
    if rr is None:
        return "layout"
    img, boxes, tboxes, sc, pairs, answer, letter_of, number_of = rr
    aud = puzzle.audit(img, tboxes, sc, dp, meta=smeta)
    bad = _gates(dp, aud, smeta)
    if bad:
        return "gate:" + bad[0]
    assert img.size == (dp["W"], dp["H"])
    return dict(img=img, boxes=boxes, tboxes=tboxes, sc=sc, pairs=pairs, answer=answer,
                letter_of=letter_of, number_of=number_of, aud=aud, smeta=smeta,
                idxs=idxs, refs=refs, dp=dp, names=names)


def _question(dp):
    return puzzle.QUESTION.format(rows=dp["ROWS"], cols=dp["COLS"], cells=dp["ROWS"] * dp["COLS"],
                             L=dp["L"], last=chr(ord("A") + dp["L"] - 1), N=dp["NSHAD"],
                             U=dp["NSHAD"] - dp["L"])


def _r(x, k=4):
    return round(float(x), k)


def build(tier, seed, *, split, index):
    if tier not in TIER_DEFINITIONS:
        raise ValueError(f"unknown tier {tier!r}")
    tab = _tables()
    reasons = collections.Counter()
    _SEARCH.clear()
    got = None
    for attempt in range(BUDGET[tier]):
        s = _sub_seed(seed, attempt)
        r = _candidate(tier, s)
        if isinstance(r, dict):
            got = r
            break
        reasons[r] += 1
    if got is None:
        raise api.Reject(f"no gate-passing seed in {BUDGET[tier]} attempts "
                         f"(top: {reasons.most_common(2)})")
    dp, aud, smeta = got["dp"], got["aud"], got["smeta"]
    L, NSHAD = dp["L"], dp["NSHAD"]
    ents, refs = tab["ents"], got["refs"]
    letter_of, number_of = got["letter_of"], got["number_of"]

    # ---- base icons and their roles: index k < L is letter_of[k]'s pattern,
    # k == L is the answerless family's base, k == L+1 the lone shadow.
    roles = [f"pattern {letter_of[k]}" for k in range(L)] + ["answerless family", "lone shadow"]
    if tier in LAW_TIERS:
        # orientation: ORIENT_OPS code applied to the upright tile (0 upright); mirrored: the orientation
        # is a reflection; silhouette_class: the orientation-free identity class the key is built from
        base = [dict(path=ents[ref[0]]["path"], name=ents[ref[0]]["name"], set=ents[ref[0]]["set"],
                     orientation=int(ref[1]), mirrored=int(ref[1]) in (1, 2, 6, 7), role=roles[k],
                     silhouette_class=got["names"][k])
                for k, ref in enumerate(refs)]
    else:
        base = [dict(path=ents[ref[0]]["path"], name=ents[ref[0]]["name"], set=ents[ref[0]]["set"],
                     mirrored=bool(ref[1]), role=roles[k],
                     silhouette_class=tab["twin_rep"].get(ents[ref[0]]["path"], ents[ref[0]]["path"]))
                for k, ref in enumerate(refs)]
    # ---- every grid cell in reading order, with what was drawn there
    cells = []
    for slot in sorted(got["sc"]):
        kind, k = got["sc"][slot]
        if kind == "p":
            cells.append(dict(slot=slot, kind="pattern", label=letter_of[k], base=k))
        else:
            m = smeta[k]
            c = dict(slot=slot, kind="shadow", label=number_of[k], shadow=k,
                     family_base=(m["src"] if m["kind"] == "clone" else k))
            if m["kind"] == "clone":
                c.update(clone=True, edit_kind=m["edit_kind"], polarity=m["polarity"],
                         site=m["site"], designed_disc_px=m["disc_px"])
            else:
                c["clone"] = False
            cells.append(c)
    shadows = []
    for j, m in enumerate(smeta):
        shadows.append(dict(index=j, label=number_of[j],
                            kind=m["kind"], family_base=(m["src"] if m["kind"] == "clone" else j),
                            matches=(letter_of[j] if j < L else None)))
    meta = dict(
        canvas=[dp["W"], dp["H"]], grid=[dp["ROWS"], dp["COLS"]], tile_px=dp["TILE"],
        layout=dict(margin=puzzle.MARGIN, gap=puzzle.GAP, label_h=puzzle.LABEL_H, font="DejaVuSans-Bold.ttf",
                    font_px=15, label_offset=[6, 1], background=[255, 255, 255],
                    border=[212, 212, 212]),
        letters=L, shadows=NSHAD, unmatched=NSHAD - L, clones_per_letter=dp["C"],
        answerless_family_clones=dp["u_clones"][0], natural=dp["natural"],
        base_icons=base, cells=cells, shadow_index=shadows,
        gate=dict(
            self_iou_min=_r(min(aud["self_iou"])),
            true_pair_disc_px_max=_r(max(aud["true_pair_disc_px"]), 2),
            min_disc_radius_px=_r(aud["min_disc_radius_px"], 2),
            min_family_disc_px=_r(aud["min_family_disc_px"], 2),
            min_shadow_shadow_disc_px=_r(aud["min_shadow_shadow_disc_px"], 2),
            rival_iou_max=_r(max(aud["rival_iou"])),
            rival_c_iou_min=_r(min(aud["rival_c_iou"])),
            rival_cplus1_iou_max=_r(max(aud["rival_cplus1_iou"])),
            exhaustive_margin=_r(aud["exhaustive_margin"]),
            area_rule_hits=int(aud["area_rule_hits"]),
        ),
        search=dict(attempts=attempt + 1, win_subseed=str(_sub_seed(seed, attempt)),
                    rejects=dict(sorted(reasons.items()))),
        generator='tasks/discrimination_grouping/find_the_shadow/generator/puzzle.py',
        cache=dict(tables=tab["cache_fp"][:16], pool=tab["pool_fp"][:16]),
    )
    if tier in LAW_TIERS:
        meta["cache"]["law"] = _law()["fp"][:16]
        # redraws before building (taken / near-duplicate / pre-screen) and the exclusion this item was drawn
        # against: the cell-start snapshot (its sha256 prefix and size), so the item can be rebuilt from its cell
        meta["search"]["redraws"] = dict(sorted(_SEARCH.items()))
        meta["search"]["exclusion"] = dict(start=_EXCL_INFO["sha256_start"][:16], n_start=_EXCL_INFO["n_start"])
    # content key: the SILHOUETTES of the base icons drawn (upright/mirrored),
    # each named by its twin class, irrespective of roles, layout, labels,
    # tints, shadow colour and clone geometry -- the task is invariant to all of
    # those -- and of the tier, so the same silhouette set can never appear
    # twice anywhere in the task.
    if tier in LAW_TIERS:
        # natural-rival rungs: the orientation-free identity set (see LAW above)
        core_key = _law_key(b["silhouette_class"] for b in base)
    else:
        core_key = digest(dict(task=TASK, base=sorted(f"{b['silhouette_class']}|{int(b['mirrored'])}"
                                                      for b in base)))
    return api.Item(question=_question(dp), answer=got["answer"], image=got["img"], meta=meta,
                    core_key=core_key, answer_type="text")


# Warm the tables in the parent process so a fork pool shares them.  A missing
# cache is NOT raised here (visro list must still work); build() raises it.
if not _missing():
    try:
        _tables()
        if (Path(os.environ.get("VISRO_FTS_LAW") or (CACHE_DIR / LAW_SUBDIR)) / "law.json").is_file():
            _law()
    except Exception:  # noqa: BLE001 - surfaced again, loudly, by build()
        _TAB = None
        _LAW = None
