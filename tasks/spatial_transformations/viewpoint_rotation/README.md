# viewpoint_rotation

Which option shows the reference cube structure seen after turning about the vertical axis?

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is a letter (A-D), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Which option A-D shows the same cube structure as the top panel, with the viewpoint turned about the vertical axis by 90, 180 or 270 degrees? Some cubes may be hidden; options are scaled to fit and omit the floor grid. Answer with the letter.

Answer: `C`

## Tiers

Larger structures with more cubes; the wrong options are mirror images or small edits of the structure.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `A` |
| medium | 448x448 | `C` |
| hard | 448x448 | `C` |
| xhard | 588x588 | `D` |
| xxhard | 588x588 | `A` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#viewpoint_rotation).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks viewpoint_rotation --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks viewpoint_rotation --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
