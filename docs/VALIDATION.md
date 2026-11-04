# Validation

Each check below is a command you can run from the VisRO repository root. The
expected results are in [validation/results.json](validation/results.json); they
were obtained on Linux x86_64 with Python 3.12 and the package and
system-library versions in [validation/environment.json](validation/environment.json).

## Generate and verify

All 24 tasks, all five tiers and all four splits at seed 20261020: 480 cells and
1,700 records. Each cell has 2 items, except count_clusters (17, so that every
question pool occurs), three_d_views and view_reconstruction (8, every view and
option-count form), find_the_different and find_the_same (6, every wording) and
logic_matrix (4, every family). Prepare the find_the_shadow tables and the
Fluent pack first ([ASSETS.md](ASSETS.md)).

```sh
n() { case $1 in count_clusters) echo 17;; three_d_views|view_reconstruction) echo 8;;
      find_the_different|find_the_same) echo 6;; logic_matrix) echo 4;; *) echo 2;; esac; }
TASKS="block_delta count_3d_blocks count_clusters count_points_on_lines cube_unfold find_the_different
  find_the_same find_the_shadow flow_free jigsaw_completion logic_matrix maze metro_map mirroring_patterns
  overlap_match overlapping_glyphs paper_folding path_tracing reconstruction rotation_patterns
  tangram_reconstruction three_d_views view_reconstruction viewpoint_rotation"
for split in train val eval ood; do
  for task in $TASKS; do
    python generate.py generate --tasks $task --tiers all --n $(n $task) --seed 20261020 \
      --split $split --out out/validation/$split
  done
  python generate.py verify --root out/validation/$split --rebuild 2
done
```

Expected: every cell accepts its items; `verify` runs each task's verifier on
every record and rebuilds records from their coordinates (all 1,700 here),
finding them byte-identical: 0 problems. The splits are generated independently here, so they
are not checked against each other; `release` and `--exclude-keys-from` keep
splits disjoint.

`results.json` also lists the SHA-256 of every cell's `records.jsonl`; a record
carries the SHA-256 of its PNG, so equal digests mean identical records and
images:

```sh
python -c "
import hashlib, json, pathlib
want = json.load(open('docs/validation/results.json'))['generate_and_verify']['records_jsonl_sha256']
root = pathlib.Path('out/validation')
diff = [k for k, v in want.items() if hashlib.sha256((root / k / 'records.jsonl').read_bytes()).hexdigest() != v]
print(len(want) - len(diff), 'of', len(want), 'cells identical', diff[:5])"
```

Byte identity needs the same Python packages, fonts, icon bytes, SVG renderer
and system libraries as `environment.json`; with other versions `verify` still
applies, the digests may not. The count_clusters digests come from a Fluent
pack whose two known-gap PNGs ([ASSETS.md](ASSETS.md#fluent-emoji-for-count_clusters))
were rendered on the validation host and pinned for this run only; where those
PNGs do not match the pinned hashes, Fluent and mixed-material items are refused
and those cells differ.

## Scorer

Scoring every published `eval` and `ood` record with its own reference answer
gives 28,800 of 28,800 correct (14,400 per split). With the downloaded dataset
in `data/VisRO/`:

```sh
for split in eval ood; do
  python -c "
import json, sys
from visro.dataset import iter_records
for r in iter_records('data/VisRO', split=sys.argv[1]):
    print(json.dumps({'id': r['id'], 'prediction': r['answer']}))" $split > reference_$split.jsonl
  python -m visro score --root data/VisRO --predictions reference_$split.jsonl --split $split
done
```

The scoring rules are in [DATASET.md](DATASET.md#scoring).

## Install

From the repository root, in a fresh environment:

- Python 3.10 or newer: `python -m pip install .` installs `visro` and
  Pillow only; `visro-score --help` and `import visro.dataset, visro.scoring`
  work.
- Python 3.12: `python -m pip install -e ".[generate]"`, then
  `python generate.py list` from the repository root lists the 24 tasks.

## Limitations

- The tokenizer-filtered mode of overlapping_glyphs was not exercised (the
  checks ran without `VISRO_TOKENIZER_DIR`).
- These are bounded samples of every task, tier and split, not an exhaustive
  check of every generator branch.
