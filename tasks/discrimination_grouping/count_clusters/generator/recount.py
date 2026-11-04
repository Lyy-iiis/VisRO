"""Independent recount of Count Clusters query predicates and masks.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import ndimage


def _question_type(record):
    return record.get("question_type", record["meta"].get("question_type"))


def _section(centroid, size, rows=3, cols=3):
    return min(rows - 1, int(centroid[1] * rows / size)) * cols + min(cols - 1, int(centroid[0] * cols / size)) + 1


def _neighbors(item):
    if "overlap_neighbors" in item:
        return item["overlap_neighbors"]
    return item["neighbors"]


def _audit_overlap_masks(meta, size):
    """Rebuild full-canvas masks from persisted runs and recount the graph."""
    instances = meta["instances"]
    masks, areas = [], []
    for item in instances:
        encoded = item["mask"]
        if encoded["encoding"] != "foreground-runs-row-major":
            raise ValueError("unsupported overlap mask encoding")
        height, width = encoded["shape"]
        flat = np.zeros(height * width, dtype=bool)
        last_end = 0
        for start, length in encoded["runs"]:
            if length <= 0 or start < last_end or start + length > len(flat):
                raise ValueError("invalid overlap mask run")
            flat[start:start + length] = True
            last_end = start + length
        x, y = item["mask_offset"]
        if x < 0 or y < 0 or x + width > size or y + height > size:
            raise ValueError("overlap mask extends outside image")
        full = np.zeros((size, size), dtype=bool)
        full[y:y + height, x:x + width] = flat.reshape((height, width))
        area = int(full.sum())
        if not area or area != item["mask_area"]:
            raise ValueError("overlap mask area differs from annotation")
        masks.append(full)
        areas.append(area)
    pairs = set()
    neighbors = {item["instance_id"]: set() for item in instances}
    for i, first in enumerate(masks):
        distances = ndimage.distance_transform_edt(~first)
        for j in range(i + 1, len(masks)):
            intersection = int(np.count_nonzero(first & masks[j]))
            if intersection:
                if intersection < max(meta["overlap_min_relative"] * min(areas[i], areas[j]),
                                      meta["overlap_min_px"]):
                    raise ValueError("overlap mask pair has ambiguous small intersection")
                a, b = instances[i]["instance_id"], instances[j]["instance_id"]
                pairs.add(tuple(sorted((a, b))))
                neighbors[a].add(b)
                neighbors[b].add(a)
            elif float(distances[masks[j]].min()) < meta["clearance_px"]:
                raise ValueError("non-overlapping mask pair violates clearance")
    if pairs != {tuple(sorted(edge["ids"])) for edge in meta["overlap_graph"]}:
        raise ValueError("persisted graph differs from overlap mask intersections")
    cover = np.zeros((size, size), dtype=bool)
    for index in reversed(range(len(instances))):
        item = instances[index]
        occluded = int(np.count_nonzero(masks[index] & cover)) / areas[index]
        if occluded > meta["gamma_cap"] + 1e-9:
            raise ValueError("overlap object exceeds occlusion limit")
        if abs(occluded - item["occluded_fraction"]) > 1e-5:
            raise ValueError("overlap occlusion annotation differs from masks")
        if neighbors[item["instance_id"]] != set(_neighbors(item)):
            raise ValueError("overlap neighbor list differs from mask intersections")
        if len(neighbors[item["instance_id"]]) > meta["max_neighbors"]:
            raise ValueError("overlap object exceeds neighbor limit")
        cover |= masks[index]
    return {key for key, value in neighbors.items() if value}


def expected_target_ids(record, size):
    """Evaluate a question from primitive attributes, without using is_target."""
    meta = record["meta"]
    instances = meta["instances"]
    kind = _question_type(record)
    if kind == "total":
        return {item["instance_id"] for item in instances}
    if kind in ("shape_subset", "color_subset", "shape_color_subset"):
        target = meta["subset_target"]
        placements = meta["placements"]
        if len(placements) != len(instances):
            raise ValueError("placement and normalized instance counts differ")
        expected = set()
        for item, placement in zip(instances, placements):
            info = placement["info"]
            if kind == "shape_color_subset":
                match = info["shape"] == target["shape"] and info["color"] == target["color"]
            else:
                match = info["shape" if kind == "shape_subset" else "color"] == target
            if match:
                expected.add(item["instance_id"])
        return expected
    if kind == "striped_subset":
        sticks = meta["sticks"]
        if len(sticks) != len(instances):
            raise ValueError("stick and normalized instance counts differ")
        return {item["instance_id"] for item, stick in zip(instances, sticks)
                if stick["style"] == "candy"}
    if kind == "section_count":
        sections = meta.get("target_sections")
        section = meta.get("target_section", meta.get("section"))
        if not sections and section is None:
            match = re.search(r"(?:section|cell)\s+(\d+)", record["question"], re.I)
            if match is None:
                raise ValueError("region query has no target section")
            section = int(match.group(1))
        sections = {int(value) for value in (sections or [section])}
        rows, cols = int(meta.get("grid_rows", 3)), int(meta.get("grid_cols", 3))
        if rows < 1 or cols < 1 or any(not 1 <= value <= rows * cols for value in sections):
            raise ValueError("target section is outside the declared grid")
        return {item["instance_id"] for item in instances
                if _section(item["centroid"], size, rows, cols) in sections}
    if kind == "category_subset":
        target = meta.get("target_class_id", meta.get("target_class"))
        if not target:
            raise ValueError("category query has no target class")
        return {item["instance_id"] for item in instances if item["class_id"] == target}
    if kind == "overlap_any":
        if all("mask" in item for item in instances):
            return _audit_overlap_masks(meta, size)
        # records without masks are recounted from their neighbour annotations;
        # generated relation records include every mask
        if not all("neighbors" in item or "overlap_neighbors" in item for item in instances):
            raise ValueError("overlap instances must expose graph neighbors")
        by_id = {item["instance_id"]: item for item in instances}
        for item in instances:
            for neighbor in _neighbors(item):
                if neighbor == item["instance_id"] or neighbor not in by_id:
                    raise ValueError("invalid overlap graph edge")
                if item["instance_id"] not in _neighbors(by_id[neighbor]):
                    raise ValueError("overlap graph is not symmetric")
        return {item["instance_id"] for item in instances if _neighbors(item)}
    raise ValueError(f"unknown mixed question type: {kind!r}")
