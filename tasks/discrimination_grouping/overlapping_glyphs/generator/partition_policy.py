"""Explicit character-count and grouping policy for overlapping-glyph scenes.

Partition selection is independent of geometry, colors, materials, and query
order. Renderers should retry the selected partition when a layout is rejected.
The spatial placement of the returned groups is deliberately left to callers.
"""
from __future__ import annotations

from functools import lru_cache
import random


HARD = ((5,), (3, 2), (1, 4))
XHARD = ((6,), (3, 3), (2, 4), (1, 5))
MAX_GROUP_SIZE = 6

POLICY = {
    "version": "glyph_partition_max_group6",
    "max_total_glyphs": 8,
    "max_group_size": MAX_GROUP_SIZE,
    "easy": {"partitions": [[3]], "selection": "fixed"},
    "medium": {"partitions": [[4]], "selection": "fixed"},
    "hard": {
        "total_glyphs": 5,
        "partitions": [list(p) for p in HARD],
        "selection": "uniform_over_listed_partitions",
    },
    "xhard": {
        "total_glyphs": 6,
        "partitions": [list(p) for p in XHARD],
        "selection": "uniform_over_listed_partitions",
    },
    "xxhard": {
        "total_glyphs": [5, 6, 7, 8],
        "group_counts": [1, 2, 3, 4],
        "feasible_group_counts_by_total": {
            "5": [1, 2, 3, 4],
            "6": [1, 2, 3, 4],
            "7": [2, 3, 4],
            "8": [2, 3, 4],
        },
        "selection": [
            "uniform_total_glyphs",
            "uniform_feasible_group_count_given_total",
            "uniform_unordered_positive_partition_given_total_count_and_max_group_size",
        ],
        "partition_representation": "nondecreasing_positive_group_sizes",
    },
    "placement": "caller_places_groups_independently_of_partition_order",
    "rejection": "retry_geometry_without_resampling_selected_partition",
}


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


@lru_cache(maxsize=None, typed=True)
def partitions(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Return every partition with at most six glyphs per group.

    Each partition is a nondecreasing tuple of exactly ``count`` parts whose
    sum is ``total``, returned in lexicographic order. An impossible positive
    count or total under the per-group cap returns an empty tuple.
    """
    _positive_int(total, "total")
    _positive_int(count, "count")
    if not count <= total <= count * MAX_GROUP_SIZE:
        return ()

    def visit(remaining: int, slots: int, minimum: int):
        if slots == 1:
            if minimum <= remaining <= MAX_GROUP_SIZE:
                yield (remaining,)
            return
        for first in range(minimum, min(MAX_GROUP_SIZE, remaining // slots) + 1):
            for rest in visit(remaining - first, slots - 1, first):
                yield (first,) + rest

    return tuple(visit(total, count, 1))


def validate_partition(tier: str, groups: list[int] | tuple[int, ...]) -> None:
    """Raise ``ValueError`` unless the proposed group multiset is allowed."""
    if tier not in ("easy", "medium", "hard", "xhard", "xxhard"):
        raise ValueError(f"unknown tier: {tier!r}")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("groups must be a nonempty list or tuple")
    for value in groups:
        _positive_int(value, "group size")
        if value > MAX_GROUP_SIZE:
            raise ValueError(f"at most {MAX_GROUP_SIZE} glyphs per group are allowed; got {value}")
    total = sum(groups)
    if total > POLICY["max_total_glyphs"]:
        raise ValueError(f"at most 8 glyphs are allowed; got {total}")
    key = tuple(sorted(groups))
    if tier == "xxhard":
        valid = 5 <= total <= 8 and 1 <= len(groups) <= 4
    else:
        allowed = {"easy": ((3,),), "medium": ((4,),), "hard": HARD, "xhard": XHARD}[tier]
        valid = key in {tuple(sorted(p)) for p in allowed}
    if not valid:
        raise ValueError(f"partition {list(groups)!r} is not allowed for {tier}")


def sample_partition(tier: str, seed: int | str) -> list[int]:
    """Select a reproducible partition without assigning spatial positions."""
    rng = random.Random(seed)
    if tier == "easy":
        chosen = (3,)
    elif tier == "medium":
        chosen = (4,)
    elif tier == "hard":
        chosen = rng.choice(HARD)
    elif tier == "xhard":
        chosen = rng.choice(XHARD)
    elif tier == "xxhard":
        total = rng.randint(5, 8)
        feasible_counts = [count for count in range(1, 5) if partitions(total, count)]
        count = rng.choice(feasible_counts)
        chosen = rng.choice(partitions(total, count))
    else:
        raise ValueError(f"unknown tier: {tier!r}")
    result = list(chosen)
    validate_partition(tier, result)
    return result
