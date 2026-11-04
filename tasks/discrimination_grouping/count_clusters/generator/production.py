"""Count Clusters per-item production and validation.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text

import hashlib
import io
import json
import os
from pathlib import Path

from shared.iconlib.fluent import STYLES
from . import question_pools

SCHEMA_VERSION = "count-clusters-production"
# fixed seed namespace of every item (changing it changes every item)
_SEED_NAMESPACE = seed_text(10471768331733853211526782592099316215018447114521469334485908420145)
DEFAULT_TOTALS = {"train": 100000, "eval": 600, "val": 100, "ood": 600}
TRAIN_TIERS = ("easy", "medium", "hard", "xhard")
SIZE = 448
SOURCE_PATHS = (
    'tasks/discrimination_grouping/count_clusters/generator/production.py',
    'tasks/discrimination_grouping/count_clusters/generator/difficulty.py', 'tasks/discrimination_grouping/count_clusters/generator/question_pools.py',
    'tasks/discrimination_grouping/count_clusters/generator/scene_ink_budget.py', 'tasks/discrimination_grouping/count_clusters/generator/fluent_scenes.py',
    'tasks/discrimination_grouping/count_clusters/generator/overlap_mixed.py', 'tasks/discrimination_grouping/count_clusters/generator/recount.py',
    'tasks/discrimination_grouping/count_clusters/generator/base/scenes.py', 'shared/iconlib/__init__.py', 'shared/iconlib/fluent.py',
    'shared/iconlib/fluent_splits.py', 'shared/iconlib/fluent_manifest.json', 'shared/iconlib/fluent_splits.json',
    'shared/iconlib/fluent_catalog.json',
    "assets/fonts/DejaVuSans.ttf", "assets/fonts/DejaVuSans-Bold.ttf",
)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_fingerprint():
    """Pin code, annotation inventory, class split and rendered fonts."""
    files = {name: _sha256((_ROOT / name).read_bytes()) for name in SOURCE_PATHS}
    return {"sha256": _sha256(_json_bytes(files)), "files": files}


def verify_fluent_assets():
    """Verify all prepared PNG bytes and the fixed train/evaluation partition."""
    manifest = json.loads(question_pools.DEFAULT_MANIFEST.read_text())
    asset_root = (_ROOT / 'assets/fluent_countable')
    classes = {"train": set(), "eval": set()}
    identities = {"train": set(), "eval": set()}
    assets = {}
    for entry in manifest["entries"]:
        split = entry["split"]
        if split not in classes:
            raise ValueError(f"unexpected Fluent asset split: {split}")
        classes[split].add(entry["class_id"])
        identities[split].add(entry["identity_id"])
        if len(entry["variants"]) != 4 or {v["style"] for v in entry["variants"]} != STYLES:
            raise ValueError(f"Fluent identity does not have all four styles: {entry['identity_id']}")
        for variant in entry["variants"]:
            path = Path(variant["path"])
            if path.is_absolute() or ".." in path.parts or path.as_posix() in assets:
                raise ValueError(f"invalid or duplicate Fluent asset path: {path}")
            actual = _sha256((asset_root / path).read_bytes())
            if actual != variant["sha256"]:
                raise ValueError(f"Fluent asset checksum mismatch: {path}")
            assets[path.as_posix()] = actual
    if len(classes["train"]) != 300 or len(classes["eval"]) != 166:
        raise ValueError("Fluent partition must contain 300 train and 166 evaluation classes")
    if classes["train"] & classes["eval"] or identities["train"] & identities["eval"]:
        raise ValueError("Fluent identities or classes leak between train and evaluation")
    return {"count": len(assets), "sha256": _sha256(_json_bytes(assets)),
            "classes": {key: len(value) for key, value in classes.items()}}


def spec_for_index(index, *, split, seed=0):
    """Shard-independent schedule, with exactly five pools per tier per cycle."""
    if split not in DEFAULT_TOTALS or not isinstance(index, int) or index < 0:
        raise ValueError("expected a nonnegative index and train/eval/val/ood split")
    if split == "ood":
        difficulty, local_index = "xxhard", index
    else:
        pools = len(question_pools.POOLS)
        difficulty = TRAIN_TIERS[(index // pools) % len(TRAIN_TIERS)]
        local_index = (index // (pools * len(TRAIN_TIERS))) * pools + index % pools
    asset_split = "train" if split == "train" else "eval"
    # A separate namespace prevents identical low-index scenes across roles/tiers.
    generation_seed = int.from_bytes(hashlib.sha256(
        f"{_SEED_NAMESPACE}:{seed}:{split}:{difficulty}".encode()).digest()[:8], "big")
    spec = question_pools.spec_for_index(local_index, seed=generation_seed, difficulty=difficulty)
    spec.update(generation_index=index, local_index=local_index,
                generation_seed=generation_seed, asset_split=asset_split)
    return spec


def _atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _image_path(index, split):
    return f"images/{split}_{index:06d}_count.png"


def _provenance(index, split, seed, spec, fingerprint):
    return {"schema_version": SCHEMA_VERSION, "split": split,
            "generation_index": index, "local_index": spec["local_index"],
            "generation_seed": spec["generation_seed"], "seed": seed,
            "asset_split": spec["asset_split"], "source_fingerprint": fingerprint["sha256"]}


def generate_record(index, *, split, seed, out, fingerprint=None):
    """Render one fixed schedule slot; failures never advance its index."""
    out = Path(out)
    fingerprint = fingerprint or build_fingerprint()
    spec = spec_for_index(index, split=split, seed=seed)
    image, question, answer, cot, meta, _ = question_pools.make_problem(
        spec["local_index"], seed=spec["generation_seed"], split=spec["asset_split"],
        difficulty=spec["difficulty"], size=SIZE)
    # The production role holds xxhard out of training; the question pools also
    # support xxhard training and therefore have no such dataset-level flag.
    meta["is_ood"] = split == "ood"
    image_path = _image_path(index, split)
    if image.size != (SIZE, SIZE):
        raise ValueError("generator returned a non-448 image")
    path = out / image_path
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    _atomic_bytes(path, buffer.getvalue())
    record = {
        "id": f"count_clusters_{split}_{index:06d}",
        "task_type": "fine_grained_discrimination", "subtype": "count_clusters",
        "level": "mid", "difficulty": meta["difficulty"],
        "generation_method": meta["generation_method"], "generation_profile": meta["generation_profile"],
        "split": split, "is_ood": split == "ood", "generation_index": index,
        "question_pool": meta["question_pool"],
        "question_type": meta["question_type"], "image": image_path,
        "question": question, "answer": answer, "ans_type": "blank", "options": [],
        "cot": cot, "meta": meta,
        "production": _provenance(index, split, seed, spec, fingerprint),
    }
    validate_record(record, index, split, seed, fingerprint)
    return record


def validate_record(record, expected_index, split, seed, fingerprint):
    """Validate schedule, provenance and labels."""
    spec = spec_for_index(expected_index, split=split, seed=seed)
    expected = {"id": f"count_clusters_{split}_{expected_index:06d}", "split": split,
                "is_ood": split == "ood",
                "generation_index": expected_index, "difficulty": spec["difficulty"],
                "question_pool": spec["pool"], "question_type": spec["question_type"],
                "generation_method": spec["generation_method"],
                "generation_profile": spec["generation_profile"], "image": _image_path(expected_index, split)}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"record {expected_index}: {key} does not match fixed schedule")
    if record.get("production") != _provenance(expected_index, split, seed, spec, fingerprint):
        raise ValueError(f"record {expected_index}: provenance mismatch")
    meta = record["meta"]
    for key in ("difficulty", "question_pool", "question_type", "generation_method", "generation_profile", "is_ood"):
        if meta.get(key) != record[key]:
            raise ValueError(f"record {expected_index}: metadata {key} mismatch")
    if meta.get("asset_split") != spec["asset_split"] or meta.get("canvas_size") != [SIZE, SIZE]:
        raise ValueError(f"record {expected_index}: asset split or canvas size mismatch")
    attempts = meta.get("generation_attempt", 0)
    expected_scene_seed = f"{spec['generation_seed']}:count-mixed:{spec['asset_split']}:{spec['local_index']}:{attempts - 1}"
    if not isinstance(attempts, int) or not 1 <= attempts <= 6 or meta.get("scene_seed") != expected_scene_seed:
        raise ValueError(f"record {expected_index}: scene seed or attempt mismatch")
    instances = meta["instances"]
    targets = [item["instance_id"] for item in instances if item["is_target"]]
    if len(set(item["instance_id"] for item in instances)) != len(instances):
        raise ValueError(f"record {expected_index}: duplicate instance IDs")
    if len(targets) != int(record["answer"]) or targets != meta.get("target_ids"):
        raise ValueError(f"record {expected_index}: answer/target mismatch")
    if meta.get("n_total") != len(instances) or not spec["count_range"][0] <= len(instances) <= spec["count_range"][1]:
        raise ValueError(f"record {expected_index}: object count outside tier range")
    if spec["pool"] != "total" and not spec["target_range"][0] <= int(record["answer"]) <= spec["target_range"][1]:
        raise ValueError(f"record {expected_index}: answer outside tier range")
    for item in instances:
        if item.get("identity_id", "").startswith("fluent/") and item.get("split") != spec["asset_split"]:
            raise ValueError(f"record {expected_index}: Fluent asset split leakage")
    return spec
