# block_delta

Count the cubes added and removed between two isometric renders of the same cube stacks.

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> BEFORE (left) and AFTER (right) show the same unit-cube stacks on a floor grid; cubes were only added to or removed from stack tops. The AFTER view may be shifted or zoomed, so match stacks by floor cell. Compare heights cell by cell (empty cell = 0). How many cubes were added or removed in total? Answer with an integer.

Answer: `6`

## Tiers

easy asks for additions only; medium and hard for the total change; xhard and xxhard for both numbers, on larger and taller grids with more edits.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `2` |
| medium | 448x448 | `3` |
| hard | 448x448 | `6` |
| xhard | 588x588 | `5,5` |
| xxhard | 588x588 | `8,5` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#block_delta).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks block_delta --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks block_delta --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
