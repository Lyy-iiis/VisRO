# Reproducing generation

## One interface

All 24 tasks use the `generate.py` entrypoint (or `python -m visro` from the
repository root). Install `requirements.txt` and prepare [docs/ASSETS.md](docs/ASSETS.md).
A small run:

```sh
python generate.py generate --tasks count_points_on_lines metro_map --tiers easy --n 2 --seed 20261020 --out out/smoke
python generate.py verify --tasks count_points_on_lines metro_map --tiers easy --root out/smoke --rebuild 2
```

The per-item seed is derived from task, tier, split, base seed and candidate
index (`visro/seeding.py`), so no two tiers or splits share a random stream.
Workers consume and accept candidates in index order, so a cell does not depend
on the worker count. Each cell holds `records.jsonl` (the 12 public fields of
[docs/DATASET.md](docs/DATASET.md)), `images/<id>.png` and `cell.json`:

| `cell.json` field | Meaning |
|---|---|
| `format`, `version` | `visro-cell@2`, release `1.0.0` |
| `task`, `category`, `tier`, `split`, `seed` | the cell's coordinates |
| `requested`, `accepted`, `candidates_consumed` | counts |
| `indices` | accepted candidate index of each record, in record order |
| `keys` | the item key of each record (content digest used for de-duplication) |
| `rejects`, `errors` | reject counts by reason; number of unexpected errors |
| `verified` | whether the verifier gate ran |
| `seconds`, `repo_commit` | wall time; repository commit that generated the cell |

`python generate.py verify` rebuilds every record (or a `--sample`) from
`(task, tier, split, seed, indices[k])`, compares the question, answer, the
other public fields and the PNG bytes, and runs the task verifier on the rebuilt
item; `--rebuild N` guarantees at least N rebuilt records per cell. Item keys
and image hashes must be unique across all cells checked together.

`release` generates train, val, eval and ood under one root, in that order;
each split excludes the item keys and images of the splits before it (the same
as `generate --exclude-keys-from`).

## Tiers

easy through xhard are the in-distribution tiers; xxhard is the
out-of-distribution tier. Generators with their own split-specific material
pools (icon, glyph and colour holdouts) draw xxhard from their held-out pool.
[docs/TASKS.md](docs/TASKS.md) gives the exact parameters of every tier.

## Environment

Pixel-identical output also requires the same Python stack, fonts, icon bytes,
SVG renderer and system libraries; package pins alone do not pin the OS. The
validated environment is in `docs/validation/environment.json`.

Tokenizer-dependent filters run only when `VISRO_TOKENIZER_DIR` is set (see
docs/ASSETS.md); without it overlapping_glyphs uses its unfiltered mode.

## Published dataset

The published VisRO dataset (100,000 train, 600 eval and 600 ood items per task)
is distributed separately as shards with public ids. VisRO is a fixed release: its items come from earlier runs of the task generators and were then given
public ids and shuffled. The generator code here regenerates items of the same tasks, difficulty levels
and record schema, byte for byte for a given task, level, split and seed, but it does not re-create the
published items one for one.
