"""Read a downloaded VisRO dataset: records and images straight from the tar shards.

    from visro.dataset import iter_examples, iter_records

    for ex in iter_examples("VisRO", tasks="core6", split="eval"):
        ex.record["question"], ex.record["answer"], ex.image_bytes   # or ex.image() (PIL)

    for rec in iter_records("VisRO", tasks=["maze"], split="ood"):    # records only
        ...

The root is the dataset directory (it holds ``tasks/<category>/<task>/<split>/``).  A shard is a
WebDataset tar: for every example, in order, the member ``<key>.json`` (the record, whose ``image``
is ``<key>.png``) followed by ``<key>.png`` (the image), where ``<key>`` is the eight-digit index of
the record's ``id``.  A shard is read from its ``NNNNN.tar`` without extracting it (also as a
non-seekable stream), or from an already extracted shard directory (``NNNNN/<key>.json`` +
``NNNNN/<key>.png``).  ``tasks`` takes task names and the alias ``core6``; ``difficulty`` takes one
level or a list.  ``verify=True`` checks every image against its record's ``image_sha256`` and every
tar file against the root's ``SHA256SUMS`` when that file exists.

Standard library only (plus ``visro.registry``); PIL is imported by ``Example.image()`` alone.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable, Iterator

from visro.registry import CATEGORY_OF, CORE6, TASKS
from visro.registry import PUBLISHED_SPLITS as SPLITS, TIERS as DIFFICULTIES

TASK_ALIASES = {"core6": CORE6, "all": TASKS}


class Example:
    """One record and its PNG bytes."""

    __slots__ = ("record", "image_bytes", "shard")

    def __init__(self, record: dict, image_bytes: bytes | None, shard: str):
        self.record = record
        self.image_bytes = image_bytes
        self.shard = shard

    def image(self):
        """The image as a PIL image (opened lazily from the bytes)."""
        from PIL import Image
        return Image.open(io.BytesIO(self.image_bytes))

    def __repr__(self):
        return f"Example({self.record.get('id')!r})"


def resolve_tasks(tasks: str | Iterable[str] | None) -> tuple[str, ...]:
    """Task names in dataset order; ``None`` means all 24.  Accepts ``core6`` / ``all`` and a
    comma-separated string."""
    if tasks is None:
        return TASKS
    if isinstance(tasks, str):
        tasks = [t for t in tasks.split(",")]
    out = []
    for name in tasks:
        name = name.strip()
        if not name:
            continue
        for t in TASK_ALIASES.get(name, (name,)):
            if t not in CATEGORY_OF:
                raise ValueError(f"unknown task {t!r}; tasks are {', '.join(TASKS)} (or core6, all)")
            if t not in out:
                out.append(t)
    return tuple(sorted(out, key=TASKS.index))


def _levels(difficulty) -> frozenset | None:
    if difficulty is None:
        return None
    levels = frozenset([difficulty] if isinstance(difficulty, str) else difficulty)
    bad = levels - set(DIFFICULTIES)
    if bad:
        raise ValueError(f"unknown difficulty {sorted(bad)}; levels are {DIFFICULTIES}")
    return levels


def find_shards(root: str | Path, tasks=None, split: str = "eval") -> list[Path]:
    """The shards (tar files or extracted shard directories) of ``tasks`` and ``split``, in order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; splits are {SPLITS}")
    root = Path(root)
    shards = []
    for task in resolve_tasks(tasks):
        d = root / "tasks" / CATEGORY_OF[task] / task / split
        if not d.is_dir():
            raise FileNotFoundError(f"{d} does not exist (download {task}/{split} first)")
        found = sorted(p for p in d.iterdir() if (p.suffix == ".tar" and p.is_file())
                       or (p.is_dir() and next(p.glob("*.json"), None) is not None))
        stems = [p.stem if p.suffix == ".tar" else p.name for p in found]
        dup = {s for s in stems if stems.count(s) > 1}
        found = [p for p, s in zip(found, stems) if s not in dup or p.suffix == ".tar"]
        if not found:
            raise FileNotFoundError(f"{d} holds no shard")
        shards.extend(found)
    return shards


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sums(root: Path) -> dict:
    path = root / "SHA256SUMS"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest
    return out


def _check_image(rec: dict, data: bytes, where: str):
    if hashlib.sha256(data).hexdigest() != rec.get("image_sha256"):
        raise ValueError(f"{where}: {rec.get('image')} does not match image_sha256 of {rec.get('id')}")


def _record(raw: bytes, member: str, where: str) -> dict:
    rec = json.loads(raw)
    key = member[: -len(".json")]
    if rec.get("image") != f"{key}.png":
        raise ValueError(f"{where}: {member} names image {rec.get('image')!r}, expected {key}.png")
    return rec


def read_shard(shard: str | Path | io.IOBase, *, images: bool = True, difficulty=None,
               verify: bool = False) -> Iterator[Example]:
    """Examples of one shard: a tar path, an extracted shard directory, or a tar file object
    (seekable or a non-seekable stream).  The members must alternate ``<key>.json``, ``<key>.png``."""
    levels = _levels(difficulty)
    name = str(shard) if isinstance(shard, (str, Path)) else str(getattr(shard, "name", "<stream>"))
    if isinstance(shard, (str, Path)) and Path(shard).is_dir():
        d = Path(shard)
        for meta in sorted(d.glob("*.json")):
            rec = _record(meta.read_bytes(), meta.name, name)
            if levels is None or rec["difficulty"] in levels:
                data = (d / rec["image"]).read_bytes() if images else None
                if verify and images:
                    _check_image(rec, data, name)
                yield Example(rec, data, name)
        return
    if isinstance(shard, (str, Path)):
        tf = tarfile.open(str(shard), "r:")
    else:
        seekable = hasattr(shard, "seekable") and shard.seekable()
        tf = tarfile.open(fileobj=shard, mode="r:" if seekable else "r|")
    with tf:
        rec = None
        for member in tf:
            if rec is None:
                if not member.name.endswith(".json"):
                    raise ValueError(f"{name}: expected a <key>.json member, found {member.name}")
                rec = _record(tf.extractfile(member).read(), member.name, name)
                continue
            if member.name != rec["image"]:
                raise ValueError(f"{name}: {rec['image']} must follow its record, found {member.name}")
            if levels is None or rec["difficulty"] in levels:
                data = tf.extractfile(member).read() if images else None
                if verify and images:
                    _check_image(rec, data, name)
                yield Example(rec, data, name)
            rec = None
        if rec is not None:
            raise ValueError(f"{name}: {rec['image']} is missing")


def iter_examples(root: str | Path, tasks=None, split: str = "eval", difficulty=None, *,
                  images: bool = True, verify: bool = False) -> Iterator[Example]:
    """Examples of ``tasks`` x ``split`` (x ``difficulty``) under the dataset ``root``."""
    root = Path(root)
    sums = _sums(root) if verify else {}
    for shard in find_shards(root, tasks, split):
        if verify and shard.suffix == ".tar":
            rel = shard.relative_to(root).as_posix()
            if rel in sums and _sha256_file(shard) != sums[rel]:
                raise ValueError(f"{rel} does not match SHA256SUMS")
        yield from read_shard(shard, images=images, difficulty=difficulty, verify=verify)


def iter_records(root: str | Path, tasks=None, split: str = "eval", difficulty=None, *,
                 verify: bool = False) -> Iterator[dict]:
    """Records only (no image bytes are read unless ``verify``)."""
    for ex in iter_examples(root, tasks, split, difficulty, images=verify, verify=verify):
        yield ex.record
