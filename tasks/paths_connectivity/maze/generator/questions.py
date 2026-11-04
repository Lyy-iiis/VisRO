"""Maze question sentences, marker choice, palette sets and record builders.

Four record shapes, one per base question type:

  * tf     : "Is there a path from A to B?"  -> True / False
  * mc     : "Which start (A/B/C/D) reaches the target marker?" -> one of A..D
  * count  : "How many of A, B, C, and D can reach the target?" -> 0..4
  * closer : "Which start is closer to the target by path length?" -> A / B

Solution-path points are emitted in image pixel space for point-trace
supervision.
"""
from __future__ import annotations

import random

from . import lattice


THICKNESSES = ("thin", "medium", "thick")


# Per-lattice marker enlargement passed to lattice.render_case. The disc radius
# is 0.42*cell*scale, so the scale is capped to keep the disc INSIDE its cell and
# off the walls: rect/polar overlap once radius passes ~0.5*cell (scale ~1.2),
# while the hexagon's inradius leaves a little more room. These are the largest
# values that stay clear of the maze walls.
MARKER_SCALE = {"rect": 1.1, "hex": 1.45, "polar": 1.15}
EXIT_MARKER_SCALE = 0.65
TARGET_MARKER_SCALES = {
    # Square targets can visually hide nearby wall openings, so keep them small.
    "square": 0.65,
    # Diamond/star occupy less solid area than a square and need a larger radius
    # to stay visible after the model-side resize to 448.
    "diamond": 0.95,
    "star": 1.05,
}


def answer_text(record: dict) -> str:
    if record.get("ansType") == "boolean":
        return "True" if record["answer"] else "False"
    return str(record["answer"])


def add_eval_qa(record: dict) -> None:
    text = answer_text(record)
    record["answer_text"] = text
    if record.get("ansType") == "boolean":
        record["qa"] = {
            "type": "maze_reachability",
            "answer_text": text,
            "choices": ["True", "False"],
        }
    elif record.get("ansType") == "number":
        record["qa"] = {
            "type": "count_reachable",
            "answer_text": text,
            "choices": record.get("options", ["0", "1", "2", "3", "4"]),
        }
    elif record.get("task_type") == "maze_closer_path":
        record["qa"] = {
            "type": "closer_path",
            "answer_text": text,
            "choices": record.get("options", ["A", "B"]),
        }
    else:
        record["qa"] = {
            "type": "maze_multi_start",
            "answer_text": text,
            "choices": record.get("options", ["A", "B", "C", "D"]),
        }


# --------------------------------------------------------------------------- #
# Question phrasing templates (language-side diversity)
# --------------------------------------------------------------------------- #
# `tf` reachability — every phrasing maps to the same boolean answer. {a}/{b}
# are the two marker labels (A and B).
TF_QUESTION_TEMPLATES = [
    "Is there a path from {a} to {b} in the maze? Answer True or False.",
    "Can you travel from {a} to {b} through the open corridors? Respond with True or False.",
    "Starting at {a}, is {b} reachable without crossing any wall? Answer True or False.",
    "Does a valid route connect {a} and {b}? Answer True or False.",
    "Determine whether {a} and {b} are connected in this maze. Answer True or False.",
    "Are points {a} and {b} linked by an open path? State True or False.",
    "Is {b} reachable from {a}? Reply with only True or False.",
    "Decide whether there is a feasible way to get from {a} to {b}. "
    "Answer with a single word: True or False.",
]

# `mc` multi-start — {opts} is the option list, e.g. "A, B, C, or D". The goal
# is an unlabeled target marker whose shape/color may vary. Every phrasing maps
# to the same single-letter answer.
MC_QUESTION_TEMPLATES = [
    "Which start ({opts}) connects to {target}? Exactly one does.",
    "Only one of the entrances {opts} reaches {target}. Which one?",
    "From which labeled start can you reach {target}? Choose {opts}.",
    "Pick the start ({opts}) that has a path to {target}.",
    "Exactly one of {opts} leads to {target}. Which letter?",
    "Which entrance — {opts} — connects to {target}? Answer with a single letter.",
    "Follow the corridors from each start: which one ({opts}) actually reaches {target}?",
    "One of {opts} can reach {target} and the others are blocked. Which start works?",
]


def _oxford_or(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + ", or " + labels[-1]


def tf_question(rng: random.Random) -> tuple[str, int]:
    i = rng.randrange(len(TF_QUESTION_TEMPLATES))
    return TF_QUESTION_TEMPLATES[i].format(a="A", b="B"), i


def mc_question(rng: random.Random, labels: list[str], target_desc: str) -> tuple[str, int]:
    i = rng.randrange(len(MC_QUESTION_TEMPLATES))
    return MC_QUESTION_TEMPLATES[i].format(opts=_oxford_or(labels), target=target_desc), i


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def points_for(lat: lattice.Lattice, path: list, dx: int, dy: int) -> list[dict]:
    out = []
    for c in path:
        px, py = lat.center(c)
        out.append({"cell": list(c), "point": [round(px - dx), round(py - dy)]})
    return out


def tf_record(idx, lat, kind, algorithm, case, dx, dy, image,
              question="Is there a path from A to B in the maze? Answer True or False.",
              qa_template=0) -> dict:
    return {
        "id": f"{kind}_tf_maze_{idx:06d}",
        "task_type": "maze_reachability",
        "lattice": kind,
        "generator": algorithm,
        "image": image,
        "question": question,
        "qa_template": qa_template,
        "ansType": "boolean",
        "answer": bool(case["solvable"]),
        "points": {"A": list(case["a"]), "B": list(case["b"])},
        "path_cells": [list(c) for c in case["path"]],
        "path_length_moves": max(0, len(case["path"]) - 1),
        "visual_primitives": {
            "type": "point_trace", "coordinate_space": "image_pixels",
            "points": points_for(lat, case["path"], dx, dy),
        },
        "maze": {"lattice": kind, "width": lat.width, "height": lat.height,
                 "image_size": list(lat.image_size())},
    }


def mc_record(idx, lat, kind, algorithm, case, dx, dy, image,
              question="Which start (A, B, C, or D) connects to the unlabeled target marker? Exactly one does.",
              qa_template=0) -> dict:
    options = [{"label": lab, "cell": list(cell)} for lab, cell in case["starts"]]
    return {
        "id": f"{kind}_mc_maze_{idx:06d}",
        "task_type": "maze_multi_start",
        "lattice": kind,
        "generator": algorithm,
        "image": image,
        "question": question,
        "qa_template": qa_template,
        "ansType": "choice",
        "options": [o["label"] for o in options],
        "option_cells": options,
        "exit": list(case["exit"]),
        "target": list(case["exit"]),
        "target_shape": case.get("target_shape", "square"),
        "answer": case["answer"],
        "path_cells": [list(c) for c in case["path"]],
        "visual_primitives": {
            "type": "point_trace", "coordinate_space": "image_pixels",
            "points": points_for(lat, case["path"], dx, dy),
        },
        "maze": {"lattice": kind, "width": lat.width, "height": lat.height,
                 "image_size": list(lat.image_size())},
    }


COUNT_QUESTION_TEMPLATES = (
    "How many of A, B, C, and D can reach {target}? Answer with just the number.",
    "How many labeled starts can get to {target}? Answer with just the number from 0 to 4.",
)
CLOSER_QUESTION_TEMPLATES = (
    "Which start is closer to {target} by path length, A or B? Answer A or B.",
    "By corridor distance, which start is closer to {target}: A or B? Answer A or B.",
)
COUNT_QUESTION = COUNT_QUESTION_TEMPLATES[0].format(target="the unlabeled target marker")
CLOSER_QUESTION = CLOSER_QUESTION_TEMPLATES[0].format(target="the unlabeled target marker")


def count_record(idx, lat, kind, algorithm, case, dx, dy, image) -> dict:
    options = [{"label": lab, "cell": list(cell)} for lab, cell in case["starts"]]
    return {
        "id": f"{kind}_count_maze_{idx:06d}",
        "task_type": "maze_count_reachable",
        "lattice": kind,
        "generator": algorithm,
        "image": image,
        "question": COUNT_QUESTION,
        "qa_template": 0,
        "ansType": "number",
        "options": ["0", "1", "2", "3", "4"],
        "option_cells": options,
        "exit": list(case["exit"]),
        "target": list(case["exit"]),
        "target_shape": case.get("target_shape", "square"),
        "answer": str(case["answer"]),
        "path_cells": [list(c) for c in case["path"]],
        "visual_primitives": {
            "type": "point_trace", "coordinate_space": "image_pixels",
            "points": points_for(lat, case["path"], dx, dy),
        },
        "maze": {"lattice": kind, "width": lat.width, "height": lat.height,
                 "image_size": list(lat.image_size())},
    }


def closer_record(idx, lat, kind, algorithm, case, dx, dy, image) -> dict:
    options = [{"label": lab, "cell": list(cell)} for lab, cell in case["starts"]]
    return {
        "id": f"{kind}_closer_maze_{idx:06d}",
        "task_type": "maze_closer_path",
        "lattice": kind,
        "generator": algorithm,
        "image": image,
        "question": CLOSER_QUESTION,
        "qa_template": 0,
        "ansType": "choice",
        "options": ["A", "B"],
        "option_cells": options,
        "exit": list(case["exit"]),
        "target": list(case["exit"]),
        "target_shape": case.get("target_shape", "square"),
        "answer": case["answer"],
        "path_cells": [list(c) for c in case["path"]],
        "visual_primitives": {
            "type": "point_trace", "coordinate_space": "image_pixels",
            "points": points_for(lat, case["path"], dx, dy),
        },
        "maze": {"lattice": kind, "width": lat.width, "height": lat.height,
                 "image_size": list(lat.image_size())},
    }


# Distinct, label-legible marker colors (white text reads on all of these; each
# contrasts with the cream background and dark walls). Used to vary marker
# colors per problem so color is not a fixed cue.
MARKER_PALETTE = [
    (52, 168, 83),    # green
    (219, 68, 55),    # red
    (244, 140, 30),   # orange
    (66, 103, 210),   # blue
    (149, 76, 233),   # purple
    (0, 150, 170),    # teal
    (200, 60, 150),   # magenta
    (140, 110, 40),   # olive
    (190, 90, 50),    # brown
    (90, 90, 90),     # slate
]

TARGET_SHAPES = ("square", "diamond", "star")
TARGET_PROMPT_STYLES = ("generic", "explicit")


def pick_marker_colors(task: str, rng: random.Random) -> dict:
    """Per-problem marker colors. tf: two distinct colors for A/B. mc: one color
    for the interchangeable start options plus a distinct target color."""
    if task == "tf":
        a, b = rng.sample(MARKER_PALETTE, 2)
        return {"a": a, "b": b}
    start, target_col = rng.sample(MARKER_PALETTE, 2)
    return {"start": start, "exit": target_col, "target": target_col}


def pick_target_shape(rng: random.Random) -> str:
    return rng.choice(TARGET_SHAPES)


def target_marker_scale(shape: str) -> float:
    return TARGET_MARKER_SCALES.get(shape, EXIT_MARKER_SCALE)


def target_descriptor(shape: str, prompt_style: str) -> str:
    if prompt_style == "generic":
        return "the unlabeled target marker"
    if shape == "square":
        return "the square marker"
    if shape == "diamond":
        return "the diamond"
    if shape == "star":
        return "the star"
    raise ValueError(f"unknown target shape: {shape!r}")


def count_question(rng: random.Random, target_desc: str) -> tuple[str, int]:
    i = rng.randrange(len(COUNT_QUESTION_TEMPLATES))
    return COUNT_QUESTION_TEMPLATES[i].format(target=target_desc), i


def closer_question(rng: random.Random, target_desc: str) -> tuple[str, int]:
    i = rng.randrange(len(CLOSER_QUESTION_TEMPLATES))
    return CLOSER_QUESTION_TEMPLATES[i].format(target=target_desc), i


def markers_for_case(task: str, case: dict, colors: dict, target_shape: str) -> list[tuple]:
    if task == "tf":
        return [(case["a"], "A", colors["a"]), (case["b"], "B", colors["b"])]
    markers = [(cell, label, colors["start"]) for label, cell in case["starts"]]
    markers.append((case["exit"], "", colors["exit"], target_shape, target_marker_scale(target_shape)))
    return markers


def resolve_palettes(split: str) -> tuple[str, ...]:
    """The palette set of a split: held-out palettes for eval, else the train ones."""
    return tuple(lattice.HELDOUT_PALETTES if split == "eval" else lattice.PALETTES)


def palette_source(palette: str) -> str:
    if palette in lattice.PALETTES:
        return "train"
    if palette in lattice.HELDOUT_PALETTES:
        return "eval"
    return "custom"
