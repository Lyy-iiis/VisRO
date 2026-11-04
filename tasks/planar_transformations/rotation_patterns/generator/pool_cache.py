"""Where the rotation_patterns pool builders write rebuilt word pools.

The shipped pools in pools/ are read first, so POOL_DIR is only used when a
pool has to be rebuilt.  VISRO_ROTATION_POOLS overrides the default
~/.cache/visro/rotation_patterns.
"""
from __future__ import annotations
import os

POOL_DIR = os.path.join(os.path.expanduser(os.environ.get(
    "VISRO_ROTATION_POOLS", os.path.join("~", ".cache", "visro", "rotation_patterns"))), "pools")
