"""Re-check finished cells from disk.

Four layers, each independent of how the cell was produced:
  contract     the cell holds only records.jsonl, images/ and cell.json; records
               parse, exactly the PUBLIC_FIELDS in order, valid values, image
               paths inside the cell, ids unique and matching cell.json
  duplicates   item keys and image sha256 unique within the cell and across
               every cell checked together (so across tiers and splits)
  rebuild      records rebuilt from (task, tier, split, seed, index) via
               cell.json must reproduce the written record and the same PNG bytes
  verifier     the task's independent correctness + uniqueness verifier on every
               rebuilt record (or a sample), against the PNG decoded from disk
"""
from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

from visro.api import PUBLIC_ANSWER_TYPES, PUBLIC_FIELDS, load_adapter, load_verifier
from visro.driver import build_record, load_cell, record_id
from visro.registry import SPLITS, TASKS, TIERS, category

CELL_ENTRIES = {"records.jsonl", "images", "cell.json"}
CELL_FIELDS = ("format", "version", "task", "category", "tier", "split", "seed", "requested", "accepted",
               "candidates_consumed", "indices", "keys", "rejects", "errors", "verified", "seconds",
               "repo_commit")


def iter_records(cell: Path):
    with open(cell / "records.jsonl") as fh:
        for n, line in enumerate(fh, 1):
            yield n, json.loads(line)


def _cell_name(cell: Path):
    for task in TASKS:
        for tier in TIERS:
            if cell.name == f"{task}_{tier}":
                return task, tier
    return None, None


def check_contract(cell: Path) -> list[str]:
    problems = []
    name = cell.name
    if not (cell / "records.jsonl").exists():
        return [f"{name}: no records.jsonl"]
    extra = sorted(p.name for p in cell.iterdir() if p.name not in CELL_ENTRIES)
    if extra:
        problems.append(f"{name}: unexpected entries {extra} (a cell holds only {sorted(CELL_ENTRIES)})")
    try:
        meta = load_cell(cell)
    except (OSError, ValueError) as exc:
        return problems + [f"{name}: cell.json unreadable: {exc}"]
    if tuple(meta) != CELL_FIELDS:
        problems.append(f"{name}: cell.json fields {list(meta)} != {list(CELL_FIELDS)}")
    task, tier = _cell_name(cell)
    if task is None or meta.get("task") != task or meta.get("tier") != tier:
        problems.append(f"{name}: cell.json task/tier do not match the cell name")
    recs = [r for _, r in iter_records(cell)]
    indices, keys = meta.get("indices") or [], meta.get("keys") or []
    if not (len(recs) == len(indices) == len(keys) == meta.get("accepted")):
        problems.append(f"{name}: {len(recs)} records, cell.json lists {len(indices)} indices / {len(keys)} keys / "
                        f"accepted={meta.get('accepted')}")
    if indices != sorted(set(indices)):
        problems.append(f"{name}: cell.json indices not strictly increasing")
    ids = set()
    for n, rec in enumerate(recs, 1):
        where = f"{name}:{n}"
        if tuple(rec) != PUBLIC_FIELDS:
            problems.append(f"{where}: fields {list(rec)} != {list(PUBLIC_FIELDS)}")
            continue
        if rec["task"] != task or rec["difficulty"] != tier:
            problems.append(f"{where}: task/difficulty {rec['task']}/{rec['difficulty']} do not match cell {name}")
        if rec["task"] not in TASKS or rec["category"] != category(rec["task"]):
            problems.append(f"{where}: bad category {rec['category']!r}")
        if rec["split"] not in SPLITS or rec["split"] != meta.get("split"):
            problems.append(f"{where}: bad split {rec['split']!r}")
        if n <= len(indices) and rec["id"] != record_id(task, tier, meta.get("split"), meta.get("seed"), indices[n - 1]):
            problems.append(f"{where}: id {rec['id']} does not match cell.json index {indices[n - 1]}")
        if not isinstance(rec["question"], str) or not rec["question"].strip():
            problems.append(f"{where}: empty question")
        if not isinstance(rec["answer"], str) or not rec["answer"].strip():
            problems.append(f"{where}: answer not a non-empty string")
        if rec["answer_type"] not in PUBLIC_ANSWER_TYPES:
            problems.append(f"{where}: bad answer_type {rec['answer_type']!r}")
        options = rec["options"]
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            problems.append(f"{where}: options must be a list of strings")
        elif rec["answer_type"] == "choice" and rec["answer"] not in options:
            problems.append(f"{where}: choice answer not in options")
        size = rec["image_size"]
        if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size)):
            problems.append(f"{where}: bad image_size {size!r}")
        if rec["id"] in ids:
            problems.append(f"{where}: duplicate id {rec['id']}")
        ids.add(rec["id"])
        if rec["image"] != f"images/{rec['id']}.png":
            problems.append(f"{where}: image path {rec['image']!r} is not images/<id>.png")
        elif not (cell / rec["image"]).is_file():
            problems.append(f"{where}: image file missing {rec['image']}")
    return problems


def rebuild(task: str, adapter, meta: dict, k: int):
    """Record k of a cell, rebuilt from cell.json: (full in-memory record, public record, png bytes)."""
    return build_record(adapter, task, meta["tier"], meta["split"], meta["seed"], meta["indices"][k])


def check_cells(cells: list[Path], *, verify_sample: int | None = None, rebuild_sample: int = 3,
                seed: int = 0) -> dict:
    """Run all four layers.  Every record is rebuilt and verified (verify_sample=None), or a sample of
    verify_sample records; at least rebuild_sample records per cell are rebuilt and compared."""
    from PIL import Image

    report = {"cells": {}, "problems": []}
    keys: dict[str, str] = {}
    shas: dict[str, str] = {}
    multi_root = len({Path(c).parent for c in cells}) > 1
    for cell in cells:
        cell = Path(cell)
        label = f"{cell.parent.name}/{cell.name}" if multi_root else cell.name
        probs = check_contract(cell)
        recs = [r for _, r in iter_records(cell)] if (cell / "records.jsonl").exists() else []
        n_ver = n_reb = 0
        usable = not probs and recs
        if usable:
            meta = load_cell(cell)
            task = meta["task"]
            verifier = load_verifier(task)
            adapter = load_adapter(task)
            rng = random.Random(f"{seed}/{cell.name}")
            for k, rec in enumerate(recs):
                key = meta["keys"][k]
                if key in keys:
                    probs.append(f"{cell.name}: {rec['id']} repeats the item of {keys[key]}")
                keys.setdefault(key, rec["id"])
                png = (cell / rec["image"]).read_bytes()
                sha = hashlib.sha256(png).hexdigest()
                if rec["image_sha256"] != sha:
                    probs.append(f"{cell.name}: {rec['id']} image bytes changed since generation")
                if sha in shas:
                    probs.append(f"{cell.name}: {rec['id']} image identical to {shas[sha]}")
                shas.setdefault(sha, rec["id"])
                with Image.open(io.BytesIO(png)) as im:
                    if list(im.size) != rec["image_size"]:
                        probs.append(f"{cell.name}: {rec['id']} image_size differs from the PNG")
            positions = list(range(len(recs)))
            chosen = set(positions if verify_sample is None else rng.sample(positions, min(verify_sample, len(recs))))
            extra = [k for k in positions if k not in chosen]
            chosen |= set(rng.sample(extra, max(0, min(rebuild_sample, len(recs)) - len(chosen))))
            for k in sorted(chosen):
                rec = recs[k]
                try:
                    full, again, png = rebuild(task, adapter, meta, k)
                except Exception as exc:  # noqa: BLE001
                    probs.append(f"{cell.name}: {rec['id']} rebuild failed: {exc!r}")
                    continue
                n_reb += 1
                if again != rec:
                    diff = [f for f in rec if rec.get(f) != again.get(f)]
                    probs.append(f"{cell.name}: {rec['id']} rebuild gives a different record (fields {diff})")
                if png != (cell / rec["image"]).read_bytes():
                    probs.append(f"{cell.name}: {rec['id']} rebuild gives different PNG bytes")
                if full["core_key"] != meta["keys"][k]:
                    probs.append(f"{cell.name}: {rec['id']} rebuild gives a different item key")
                im = Image.open(cell / rec["image"]).convert("RGB")
                try:
                    for p in verifier.verify(full, im):
                        probs.append(f"{cell.name}: {rec['id']} verifier: {p}")
                except Exception as exc:  # noqa: BLE001
                    probs.append(f"{cell.name}: {rec['id']} verifier crashed: {exc!r}")
                n_ver += 1
        report["cells"][label] = {"records": len(recs), "verified": n_ver, "rebuilt": n_reb,
                                  "problems": len(probs)}
        report["problems"] += probs
    return report
