# view_reconstruction

Which isometric cube structure matches two or three given orthographic views?

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is a letter (A-D), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> The top row shows the top view from above with the back row at the top, front view from the front, and side view from the right with the front row on the left; a square is filled if any cube is in its line of sight, floor at the bottom. Options A-D show cube structures on the full floor grid, FRONT faces lower left and SIDE faces lower right. Which option matches all three views? Answer with its letter.

Answer: `D`

## Tiers

easy and medium give two views, hard to xxhard all three, on larger structures.

| tier | image (px) | example answer |
|---|---|---|
| easy | 588x588 | `D` |
| medium | 448x448 | `A` |
| hard | 448x448 | `D` |
| xhard | 588x588 | `B` |
| xxhard | 588x588 | `D` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#view_reconstruction).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks view_reconstruction --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks view_reconstruction --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
