"""Generate one (task, tier) cell: build -> independent verify -> dedup -> write.

Cell layout:

    <out>/<task>_<tier>/records.jsonl      one JSON record per line (the PUBLIC_FIELDS, in order)
    <out>/<task>_<tier>/images/<id>.png    question images, paths relative to the cell
    <out>/<task>_<tier>/cell.json          what was generated, and reject statistics

A finished cell holds exactly these three entries (`visro verify` checks it).

Candidates are consumed in index order and accepted in index order whatever
the worker count, so a cell is a pure function of (task, tier, split, seed, n)
and the code.  cell.json lists the accepted candidate indices in record order,
so every record can be rebuilt and re-verified (`visro verify`).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import io
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from pathlib import Path

from visro.api import PUBLIC_FIELDS, RELEASE_VERSION, REJECT_EXCEPTIONS, load_adapter, load_verifier
from visro.paths import ROOT
from visro.registry import SPLITS, TIERS, category
from visro.seeding import derive_seed, plain

FORMAT = "visro-cell@2"
ERROR_LIMIT = 5

_ADAPTER = None  # set by _init before the pool forks
_VERIFIER = None


def repo_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - best effort outside a checkout
        return "unknown"


def record_id(task: str, tier: str, split: str, seed: int, index: int) -> str:
    return f"{task}_{tier}_{split}_s{seed}_{index:07d}"


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_record(task, tier, split, seed, index, item, adapter_version) -> dict:
    """The full in-memory record the verifier checks.  Only public_record(...) of it is written."""
    rid = record_id(task, tier, split, seed, index)
    rec = {
        "id": rid,
        "task": task,
        "tier": tier,
        "difficulty": tier,
        "split": split,
        "question": item.question,
        "image": f"images/{rid}.png",
        "answer": item.answer,
        "answer_type": item.answer_type,
    }
    if item.choices is not None:
        rec["choices"] = list(item.choices)
    rec["core_key"] = item.core_key
    rec["meta"] = plain(item.meta)
    rec["source"] = {"task_version": adapter_version, "seed": int(seed), "index": int(index),
                     "derived_seed": derive_seed(task, tier, split, seed, index)}
    return rec


def public_record(rec: dict, item, png: bytes) -> dict:
    """The written record: PUBLIC_FIELDS, in that order, the same for every task."""
    out = {
        "id": rec["id"],
        "task": rec["task"],
        "category": category(rec["task"]),
        "split": rec["split"],
        "difficulty": rec["tier"],
        "question": rec["question"],
        "answer": rec["answer"],
        "answer_type": item.public_answer_type(),
        "options": item.public_options(),
        "image": rec["image"],
        "image_sha256": hashlib.sha256(png).hexdigest(),
        "image_size": [int(v) for v in item.image.size],
    }
    assert tuple(out) == PUBLIC_FIELDS
    return out


def build_record(adapter, task: str, tier: str, split: str, seed: int, index: int):
    """Candidate `index` of a cell: (full in-memory record, public record, png bytes).  Raises whatever
    the adapter raises.  Generation and `visro verify` both build through here."""
    item = adapter.build(tier, derive_seed(task, tier, split, seed, index), split=split, index=index)
    item.validate()
    rec = make_record(task, tier, split, seed, index, item, adapter.VERSION)
    json.dumps(rec, allow_nan=False)  # must serialise
    png = png_bytes(item.image)
    return rec, public_record(rec, item, png), png


def build_one(task: str, tier: str, split: str, seed: int, index: int, verify: bool = True):
    """Returns ("ok", record, public record, png) | ("reject", reason) | ("error", traceback)."""
    from PIL import Image

    try:
        rec, pub, png = build_record(_ADAPTER, task, tier, split, seed, index)
    except REJECT_EXCEPTIONS as exc:
        return ("reject", "build: " + _short(exc))
    except Exception:  # noqa: BLE001
        return ("error", traceback.format_exc())
    if verify:
        try:
            problems = _VERIFIER.verify(rec, Image.open(io.BytesIO(png)).convert("RGB"))
        except Exception:  # noqa: BLE001 - a crashing verifier is a bug, not a reject
            return ("error", "verifier crashed:\n" + traceback.format_exc())
        if problems:
            return ("reject", "verify: " + _short(problems[0]))
    return ("ok", rec, pub, png)


def _short(x) -> str:
    s = str(x).strip().splitlines()[0] if str(x).strip() else type(x).__name__
    return s[:120]


def _init(task: str, verify: bool):
    global _ADAPTER, _VERIFIER
    _ADAPTER = load_adapter(task)
    _VERIFIER = load_verifier(task) if verify else None


def _work(args):
    return build_one(*args)


def generate_cell(task: str, tier: str, out_root: Path, *, n: int, seed: int, split: str = "train",
                  workers: int = 1, max_candidates: int | None = None, seen: set | None = None,
                  verify: bool = True, overwrite: bool = False, log=print, prebuild=None) -> dict:
    assert tier in TIERS and split in SPLITS, (tier, split)
    out_root = Path(out_root)
    cell = out_root / f"{task}_{tier}"
    if (cell / "records.jsonl").exists() and not overwrite:
        raise FileExistsError(f"{cell}/records.jsonl exists (use --overwrite)")
    (cell / "images").mkdir(parents=True, exist_ok=True)
    for stale in list(cell.glob("records*.jsonl")):
        stale.unlink()
    max_candidates = max_candidates or (n * 20 + 50)
    seen = seen if seen is not None else set()
    _init(task, verify)
    # an adapter that can tell before building that an attempt would draw a key already taken gets the
    # seen set as it stands at the start of the cell (a snapshot: an item then depends on its seed and on
    # the earlier cells, never on worker timing), plus any further keys a caller rejects after building
    # (`prebuild`, e.g. a release runner's near-duplicate rule keys); every other adapter is unaffected
    if hasattr(_ADAPTER, "prebuild_exclusion"):
        _ADAPTER.prebuild_exclusion(frozenset(seen) | frozenset(prebuild or ()))

    reasons: collections.Counter = collections.Counter()
    errors: list[str] = []
    accepted: list[dict] = []
    indices: list[int] = []
    keys: list[str] = []
    t0 = time.time()
    tmp = cell / "records.jsonl.tmp"
    consumed = 0
    with open(tmp, "w") as fh:
        def accept(res, index):
            _, rec, pub, png = res
            sha = pub["image_sha256"]
            if rec["core_key"] in seen:
                reasons["duplicate item"] += 1
                return
            if "png:" + sha in seen:
                reasons["duplicate image"] += 1
                return
            seen.add(rec["core_key"])
            seen.add("png:" + sha)
            (cell / pub["image"]).write_bytes(png)
            fh.write(json.dumps(pub, ensure_ascii=False, allow_nan=False) + "\n")
            accepted.append(pub)
            indices.append(index)
            keys.append(rec["core_key"])

        def handle(res, index):
            if res[0] == "ok":
                accept(res, index)
            elif res[0] == "reject":
                reasons[res[1]] += 1
            else:
                errors.append(f"index {index}: {res[1]}")
                if len(errors) >= ERROR_LIMIT:
                    raise RuntimeError(f"{task}/{tier}: {len(errors)} unexpected errors; first:\n{errors[0]}")

        if workers <= 1:
            for index in range(max_candidates):
                if len(accepted) >= n:
                    break
                handle(build_one(task, tier, split, seed, index, verify), index)
                consumed = index + 1
        else:
            ctx = mp.get_context("fork")
            batch = max(workers * 4, 8)
            with cf.ProcessPoolExecutor(workers, mp_context=ctx) as pool:
                index = 0
                while len(accepted) < n and index < max_candidates:
                    hi = min(index + batch, max_candidates)
                    args = [(task, tier, split, seed, i, verify) for i in range(index, hi)]
                    for i, res in zip(range(index, hi), pool.map(_work, args)):
                        if len(accepted) >= n:
                            break
                        handle(res, i)
                        consumed = i + 1
                    index = hi
    tmp.replace(cell / "records.jsonl")
    stats = {
        "format": FORMAT, "version": RELEASE_VERSION,
        "task": task, "category": category(task), "tier": tier, "split": split, "seed": seed,
        "requested": n, "accepted": len(accepted), "candidates_consumed": consumed,
        # accepted candidate indices and item keys, in record order: visro verify rebuilds record k from
        # indices[k]; the keys keep later splits disjoint from this one (--exclude-keys-from)
        "indices": indices, "keys": keys,
        "rejects": dict(reasons.most_common()), "errors": len(errors), "verified": bool(verify),
        "seconds": round(time.time() - t0, 2), "repo_commit": repo_commit(),
    }
    (cell / "cell.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n")
    log(f"{task}_{tier}: {len(accepted)}/{n} accepted from {consumed} candidates "
        f"in {stats['seconds']}s; rejects={sum(reasons.values())} errors={len(errors)}")
    if len(accepted) < n:
        raise RuntimeError(f"{task}/{tier}: only {len(accepted)}/{n} accepted within {max_candidates} candidates; "
                           f"top rejects: {reasons.most_common(5)}")
    return stats


def load_cell(cell: Path) -> dict:
    return json.loads((Path(cell) / "cell.json").read_text())


def load_seen(dirs, task: str | None = None) -> set:
    """Item keys and image hashes already used by existing cells (keeps eval disjoint from train)."""
    keys = set()
    for d in dirs:
        files = Path(d).glob(f"{task}_*/records.jsonl") if task else Path(d).rglob("records.jsonl")
        for f in files:
            keys.update(load_cell(f.parent)["keys"])
            with open(f) as fh:
                for line in fh:
                    keys.add("png:" + json.loads(line)["image_sha256"])
    return keys
