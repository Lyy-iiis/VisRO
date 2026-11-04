# count_3d_blocks

Count the unit cubes in an isometric stack: the total, or the fewest cubes consistent with the view.

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> This figure of unit cubes stands on the floor grid; no cube floats, some may be hidden, and the cubes need not be connected. What is the fewest cubes that would look exactly like this from this viewpoint? Answer with an integer.

Answer: `18`

## Tiers

easy asks for the total of a figure whose every column top is visible; from medium on the question asks for the minimum, on larger bases with taller stacks.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `9` |
| medium | 448x448 | `17` |
| hard | 448x448 | `18` |
| xhard | 588x588 | `49` |
| xxhard | 588x588 | `65` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#count_3d_blocks).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks count_3d_blocks --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks count_3d_blocks --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
