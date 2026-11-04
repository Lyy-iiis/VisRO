# find_the_different

A grid of copies of one icon in which exactly one cell is altered; give its row and column.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> The image shows 48 icons in 8 rows and 6 columns, and one differs from the others. Answer its position as (row,column), counting from 1 from the top and from the left.

Answer: `(2,1)`

## Tiers

More cells (4x4 on easy to 10x10 on xxhard) and, on the larger grids, a cap on how many pixels the alteration may change.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `(5,2)` |
| medium | 448x448 | `(1,4)` |
| hard | 448x448 | `(2,1)` |
| xhard | 588x588 | `(1,3)` |
| xxhard | 588x588 | `(2,1)` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#find_the_different).

train, eval and val draw from the icon library's train icons; ood draws from the held-out test icons, minus those whose artwork is identical to a train icon.

## Assets

Only the bundled icon tree and fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks find_the_different --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks find_the_different --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
