# VisRO record format

The published VisRO dataset and the cells written by `python generate.py
generate` use the same record schema: 12 fields, in this order, for all 24 tasks.

| Field | Meaning |
|---|---|
| `id` | published: `visro24/<task>/<split>/<eight-digit index>`; generated: `<task>_<tier>_<split>_s<seed>_<seven-digit candidate index>` |
| `task`, `category` | one of the 24 tasks and its category (the `tasks/` tree) |
| `split` | `train`, `eval` or `ood` (generated cells may also be `val`) |
| `difficulty` | `easy`, `medium`, `hard`, `xhard` or `xxhard` |
| `question` | the question; it states the required answer format |
| `answer` | the reference answer, always a string |
| `answer_type` | `choice` (answer is one of `options`) or `free_form` |
| `options` | the answer options of a choice question; a free-form question may also list suggested values; empty otherwise |
| `image` | published: the image member of the shard, `<eight-digit index>.png`; generated: the PNG path in the cell, `images/<id>.png` |
| `image_sha256`, `image_size` | sha256 of the PNG bytes and `[width, height]` |

Nothing else is written per record.  Task-specific grading follows the answer
format the question states (exact strings, letter lists, bitstrings, numbers).

## Published dataset

| Split | Per task | All 24 tasks | Difficulty levels |
|---|---:|---:|---|
| train | 100,000 | 2,400,000 | easy / medium / hard / xhard, 25,000 each |
| eval | 600 | 14,400 | easy / medium / hard / xhard, 150 each |
| ood | 600 | 14,400 | xxhard |

Total: **2,428,800 examples in 528 shards**.

```text
README.md
schema.json
counts.json
SHA256SUMS
tasks/<category>/<task>/train/00000.tar ... 00019.tar
tasks/<category>/<task>/eval/00000.tar
tasks/<category>/<task>/ood/00000.tar
```

A training shard holds 5,000 examples; an eval or ood shard holds 600. The
shards are in WebDataset format: each example is two consecutive members,
`<key>.json` (its record) and `<key>.png` (its image), where `<key>` is the
eight-digit index of its `id`; a shard holds nothing else. Examples are
shuffled within a shard, so every shard mixes the difficulty levels.

VisRO is a fixed release: its items come from earlier runs of the task generators and were then given
public ids and shuffled. The generator code here regenerates items of the same tasks, difficulty levels
and record schema, byte for byte for a given task, level, split and seed, but it does not re-create the
published items one for one.

```python
import json
import tarfile

with tarfile.open("00000.tar", "r:") as shard:
    members = shard.getmembers()
    for meta, image in zip(members[0::2], members[1::2]):
        row = json.loads(shard.extractfile(meta).read())
        png_bytes = shard.extractfile(image).read()
        # row["question"], row["answer"], png_bytes
```

## Reading

`visro.dataset` (standard library; Pillow only for `Example.image()`) reads a
downloaded dataset directory without extracting it:

```python
from visro.dataset import iter_examples, iter_records

for ex in iter_examples("VisRO", tasks="core6", split="eval", difficulty=["hard", "xhard"]):
    ex.record, ex.image_bytes, ex.image()       # record dict, PNG bytes, PIL image

records = list(iter_records("VisRO", tasks=["maze", "path_tracing"], split="ood"))
```

`tasks` takes task names, `core6` (count_clusters, flow_free, maze, metro_map,
overlapping_glyphs, path_tracing) or `all`. A shard is read from its tar file,
or from a directory that holds an extracted shard (`<key>.json`, `<key>.png`).
`read_shard` also accepts a tar file object, including a non-seekable stream;
it refuses members that are not `<key>.json` followed by its `<key>.png`. With
`verify=True` every image is checked against `image_sha256` and every tar file
against `SHA256SUMS` when the root holds one.

## Scoring

`visro.scoring` (standard library) scores a prediction against a published
record. It reads only `task`, `answer_type`, `options`, `question` and `answer`.

```python
from visro.scoring import score_record, summarize
score_record(record, "B")   # {"correct": True, "parsed": ..., "gold": ..., "rule": "choice"}
```

```sh
python -m visro score --root VisRO --predictions preds.jsonl --tasks core6 --split eval \
    --per-item items.jsonl --out summary.json
```

`preds.jsonl` holds one `{"id": ..., "prediction": ...}` per line. The command
prints the accuracy per task, the macro mean over tasks and the micro mean; a
record without a prediction counts as wrong and is reported. `summary.json` also
has the accuracy per task and difficulty.

**Core6** follows the scoring rules of the paper's experiments. Three inputs of
those rules are not separate fields of the published records, so the scorer
derives them from the question text:

| Task | Derived from the question | Rule |
|---|---|---|
| maze | answer kind: `choice` (answer_type), `boolean` (the question asks for True or False), `number` | choice: the first character if it is A-F, else the first of A..F that stands alone; boolean: the first yes/no/true/false/1/0 token; number: the first integer |
| path_tracing | question type (one template each): one_marker, ordered, all_pairs, same_line; for ordered, the scene's labels: A-Z and 1 to max(9, position of the last start letter named) | one label; the label sequence (an unseparated answer must split one way only); the set of letter-number pairs; yes/no |
| metro_map | a question that asks for a station name | the name, case-insensitive; otherwise one integer |

count_clusters: exactly one integer; flow_free: the integer list in order;
overlapping_glyphs: the letters and digits in order.

**The other 18 tasks** use one strict rule per task: the whole cleaned
prediction must be one complete answer of the stated format (choice letters
that must be options; unordered pairs for find_the_same; the exact string,
whitespace removed, for count_points_on_lines, find_the_shadow and
jigsaw_completion; bit strings, grids, colour codes, figure lists and
placements in order). Cleaning removes surrounding whitespace, a leading
`answer:`, quotes, backticks, `**bold**`, `\boxed{}` and a trailing period; a
multi-line prediction is tried whole, then by its first non-empty line. The
module docstring lists every rule.

**Checks.** On all 28,800 published eval and ood records the derived inputs
agree with the original generator records, and every reference answer scores
as correct ([VALIDATION.md](VALIDATION.md#scorer) has the command).

## Generated cells

```text
<out>/<task>_<tier>/records.jsonl
<out>/<task>_<tier>/images/<id>.png
<out>/<task>_<tier>/cell.json
```

`cell.json` records the coordinates every record was built from (see
[../REPRODUCING.md](../REPRODUCING.md)); `python generate.py verify` uses it to
rebuild and re-verify the records.
