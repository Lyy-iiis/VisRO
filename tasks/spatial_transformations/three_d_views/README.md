# three_d_views

Pick the orthographic front, right-side or top view of an isometric cube stack.

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is a letter (A-H), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Which option A-D is the right-side view of this cube figure, in the direction of the arrow? Filled squares are where the figure is seen (the grid may extend past it); the bottom row is floor level. Answer with the letter.

Answer: `C`

## Tiers

Larger and taller stacks and options that differ by smaller edits; xxhard has eight options.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `B` |
| medium | 448x448 | `B` |
| hard | 448x448 | `C` |
| xhard | 588x588 | `B` |
| xxhard | 588x588 | `B` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#three_d_views).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks three_d_views --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks three_d_views --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
