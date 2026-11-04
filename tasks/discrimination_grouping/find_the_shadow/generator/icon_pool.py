"""Icon-silhouette pool of find_the_shadow: icons with their masks cached at
160x160 canonical resolution, so the XOR-difference gate can be evaluated at
the actual tile size.  The pool is the first POOL_N train icons, in
POOL_SEED-shuffled order, that render to a usable silhouette; build/build_cache.py
writes it to POOL_FILE.
"""
import pickle

import numpy as np

from .cache import BUILD_HINT, CACHE_DIR

POOL_N, POOL_SEED = 14000, 777
RENDER_S = 256          # render resolution before canonicalisation
STORE_S = 160           # canonical mask resolution kept in the cache
POOL_FILE = CACHE_DIR / "pool3.pkl"


def load():
    """(entries, packed 160x160 masks, largest-component ratios, fills) of the cached pool."""
    if not POOL_FILE.is_file():
        raise FileNotFoundError(f"find_the_shadow icon pool {POOL_FILE} is missing; " + BUILD_HINT)
    with open(POOL_FILE, "rb") as f:
        return pickle.load(f)


def unpack(packed, idx=None):
    P = packed if idx is None else packed[idx]
    n = len(P)
    return np.unpackbits(P, axis=1)[:, :STORE_S * STORE_S].reshape(
        n, STORE_S, STORE_S).astype(bool)
