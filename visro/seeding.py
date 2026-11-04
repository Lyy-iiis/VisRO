"""Per-item seeds.

Every item's random stream is keyed by (task, tier, split, base seed, index),
so no two tiers or splits share a stream.
"""
from __future__ import annotations

import hashlib
import json


def seed_text(value: int) -> str:
    """The text of a fixed RNG namespace.  Generators seed their random streams from it, so the
    value must never change: changing one changes every item drawn from that stream."""
    return value.to_bytes((value.bit_length() + 7) // 8, "big").decode("ascii")


_SEED_NAMESPACE = seed_text(8242826903332812337)


def derive_seed(task: str, tier: str, split: str, seed: int, index: int) -> int:
    key = f"{_SEED_NAMESPACE}/{task}/{tier}/{split}/{int(seed)}/{int(index)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    """sha256 of the canonical JSON of `value` -- the usual way to build core_key."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def plain(value):
    """numpy / tuple-laden structures -> JSON-safe lists, dicts and scalars."""
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value
