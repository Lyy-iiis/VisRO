"""The word pools the ladder draws from, each pinned by SPEC.

Every pool is the canonical json of a deterministic builder: SPEC records the
builder (module under build/, function), the word count and the md5 of the
canonical json.  load() takes, in this order, the shipped copy pools/<name>.gz,
the cache under $VISRO_ROTATION_POOLS/pools, or a rebuild (a cold rebuild is
10-60 min of CPU).  Every route asserts the SPEC md5 and word count, so a stale
or edited pool fails loudly instead of loading.

    pool              builder                          gates                    words  tiers
    _pool_L3w8full    exhaustive_pools.build_l3_full   W1..W8 over all 3^16     1,553  easy (1 index in 16)
    _pool_L4w8full    exhaustive_pools.build_l4_full   W1..W8 over all 4^16    34,811  easy (15 in 16)
    _pool_L5m         length5_pools.build_medium       W1..W7, W8 <= 0.17 px  126,936  medium
    _pool_L5w8        length5_pools.build_w8           W1..W8, every multiset  35,771  hard, xxhard
    _pool_L5xwd       length5_pools.build_twin_wide    d(8) in {6, 8, 10}      17,565  xhard
"""
from __future__ import annotations
import gzip, hashlib, importlib, json, os, sys

from .pool_cache import POOL_DIR

def canonical(L, cfg, max_pred, words, **extra):
    """The one serialisation every md5 in SPEC is taken over.  Sorted keys,
    no spaces, words as lists of ints - so the checksum is a function of the
    CONTENT and not of whoever wrote the file out."""
    d = dict(L=L, cfg=cfg, max_pred=max_pred,
             words=[[int(x) for x in w] for w in words])
    d.update(extra)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


# --------------------------------------------------------------------- SPEC
# md5 is over canonical(...) above.  n_words and md5 are what load() enforces;
# `build` is (module under build/, function) and the function takes no arguments - every
# constant it uses lives in that module, next to the prose that explains it.
SPEC = {}
SPEC.update({
    "_pool_L5w8.json": dict(
        kind="spec", L=5, build=("length5_pools", "build_w8"),
        n_words=35771,
        md5="ad8748ba4663adc159594366eaf8c016",
        note="W1..W8, every level multiset, fixed-seed uniform search over 5^16"),
    "_pool_L3w8full.json": dict(
        kind="spec", L=3, build=("exhaustive_pools", "build_l3_full"),
        n_words=1553,
        md5="ff47aaddb350b4d274bd1a6a6abaff9b",
        note="W1..W8 over all of 3^16, exhaustive"),
    "_pool_L5xwd.json": dict(
        kind="spec", L=5, build=("length5_pools", "build_twin_wide"),
        n_words=17565,
        md5="6a81313456eea76c93f0b23ff339ddfb",
        note="near-twin family: d(8) in {6,8,10} with W5's floor tied to each "
             "word's own d(8), so the closest pair of rings on the canvas is the "
             "180 deg twin.  A cold rebuild is ~10 min of search on 6 cores plus "
             "~50 min of P2 renders"),
    "_pool_L5m.json": dict(
        kind="spec", L=5, build=("length5_pools", "build_medium"),
        n_words=126936,
        md5="94560be3496e09a86fef44e904699672",
        note="medium: hard's L=5 W1..W7 family (word_gates.CFG[5]) "
             "ENUMERATED over all of 5^16 (173,622 necklaces at MAX_PRED 0.20, of which _pool_L5w8 is a "
             "20.6% sample), with W8 set to MED_MAX_PRED = 0.17 for this tier: every necklace whose ideal "
             "centroid is <= 0.17 px (47,823 classes, 186,857 figures).  No class excluded.  A cold "
             "rebuild is ~10 min of one core (VISRO_ROTATION_WORKERS=N: ~2 min on 7, same bytes)"),
    "_pool_L4w8full.json": dict(
        kind="spec", L=4, build=("exhaustive_pools", "build_l4_full"),
        n_words=34811,
        md5="2b69aa46d939b00416a2888936442525",
        note="W1..W8 over all of 4^16, exhaustive"),
})


SHIPPED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pools")


def _shipped_text(name):
    """The canonical json of a pool shipped as pools/<name>.gz, or None.  The
    caller asserts the SPEC md5 on it exactly as on a cached or rebuilt copy."""
    p = os.path.join(SHIPPED_DIR, name + ".gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as fh:
        return fh.read().decode("ascii")


def _rebuild(name):
    mod, fn = SPEC[name]["build"]
    print(f"word_pools: {name} is not cached; rebuilding it with "
          f"build.{mod}.{fn}() - this is slow and happens once.",
          file=sys.stderr, flush=True)
    m = importlib.import_module(f".build.{mod}", __package__)
    return getattr(m, fn)()


def load(name):
    """The parsed pool, from wherever it can be had, with the md5 enforced at
    every route."""
    sp = SPEC[name]
    t = _shipped_text(name)
    if t is None:
        cache = os.path.join(POOL_DIR, name)
        if os.path.exists(cache):
            t = open(cache).read()
            if md5(t) != sp["md5"]:
                print(f"word_pools: cached {name} has md5 {md5(t)}, SPEC "
                      f"says {sp['md5']} - discarding the cache.",
                      file=sys.stderr, flush=True)
                t = None
        if t is None:
            t = canonical(*_rebuild(name))
            os.makedirs(POOL_DIR, exist_ok=True)
            tmp = cache + ".tmp"
            open(tmp, "w").write(t)
            os.replace(tmp, cache)          # never a half-written pool on disk
    assert md5(t) == sp["md5"], (
        f"{name}: md5 {md5(t)} != SPEC {sp['md5']}: the builder no longer "
        f"reproduces the pinned pool.")
    d = json.loads(t)
    assert len(d["words"]) == sp["n_words"], (
        f"{name}: {len(d['words'])} words, SPEC says {sp['n_words']}")
    return d
