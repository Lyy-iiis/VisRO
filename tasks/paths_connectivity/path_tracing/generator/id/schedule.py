"""Training block schedule seed.
"""

from __future__ import annotations


def path_schedule_seed(master_seed: int, block: int) -> int:
    # Schedule-only seed; geometry continues to use master_seed + global index.
    return master_seed + (block + 1) * 10_000_019
