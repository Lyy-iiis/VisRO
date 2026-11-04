"""Balanced path schedule helpers.
"""

from __future__ import annotations

import random


CURVE_STYLES = ("bezier", "pathfinder")
DIFFICULTIES = ("easy", "medium", "hard", "xhard")
QA_TYPES = ("one_marker", "same_line", "ordered", "all_pairs")
LAYOUT_TARGETS = {
    "pathfinder": {
        "left_right": 0.20,
        "top_bottom": 0.20,
        "adjacent_sides": 0.10,
        "perimeter_random": 0.25,
        "scattered": 0.25,
    },
}


def _integer_layout_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def balanced_schedule(
    total: int,
    seed: int,
) -> list[tuple[str, str, str, str | None]]:
    """Return exact style, difficulty, and QA marginal quotas.

    Every style x difficulty cell has the same size. Its remainder over the four
    QA types is rotated so the global QA marginal remains exact.
    """
    style_difficulty_cells = [
        (style, difficulty)
        for style in CURVE_STYLES
        for difficulty in DIFFICULTIES
    ]
    if total % len(style_difficulty_cells):
        raise ValueError("--count must be divisible by 8 for exact balance")

    cell_total = total // len(style_difficulty_cells)
    base, remainder = divmod(cell_total, len(QA_TYPES))
    base_schedule: list[tuple[str, str, str]] = []
    for cell_index, (style, difficulty) in enumerate(style_difficulty_cells):
        counts = [base] * len(QA_TYPES)
        # This order gives every QA type exactly two extras across each complete
        # pass over the eight style/difficulty cells.
        extra_order = (
            cell_index % 4,
            (cell_index + 2) % 4,
            (cell_index + 1) % 4,
            (cell_index + 3) % 4,
        )
        for qa_index in extra_order[:remainder]:
            counts[qa_index] += 1
        for qa_type, count in zip(QA_TYPES, counts):
            base_schedule.extend((style, difficulty, qa_type) for _ in range(count))

    random.Random(seed).shuffle(base_schedule)
    # Bezier endpoints are freely spread by the engine. Pathfinder uses exact
    # quotas because its named layout families have distinct geometry.
    schedule: list[list[str | None]] = [list(item) + [None] for item in base_schedule]
    indices = [i for i, item in enumerate(schedule) if item[0] == "pathfinder"]
    layout_counts = _integer_layout_counts(
        len(indices), LAYOUT_TARGETS["pathfinder"],
    )
    layouts = [
        layout
        for layout, count in layout_counts.items()
        for _ in range(count)
    ]
    random.Random(f"{seed}:pathfinder:layouts").shuffle(layouts)
    for index, layout in zip(indices, layouts):
        schedule[index][3] = layout

    for i, item in enumerate(schedule):
        style, difficulty, qa_type, layout = item
        if (
            style == "pathfinder"
            and difficulty in {"hard", "xhard"}
            and qa_type == "all_pairs"
            and layout not in {"left_right", "top_bottom"}
        ):
            replacement = next(
                j for j, other in enumerate(schedule)
                if other[0] == "pathfinder"
                and not (
                    other[1] in {"hard", "xhard"}
                    and other[2] == "all_pairs"
                )
                and other[3] in {"left_right", "top_bottom"}
            )
            schedule[i][3], schedule[replacement][3] = (
                schedule[replacement][3], schedule[i][3]
            )

    return [tuple(item) for item in schedule]
