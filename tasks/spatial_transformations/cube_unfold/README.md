# cube_unfold

Which numbered isometric cubes are views of the cube that a net folds into?

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Fold the top net into a cube, drawn side out. Which numbered cubes, each showing its three top-front-corner faces, are this cube after some rotation, with faces, layout and symbol orientations all matching (mirror images do not)? Answer the numbers ascending, comma-separated, no spaces.

Answer: `3,8`

## Tiers

More candidates and more matches; from hard on the faces carry orientation glyphs, and on xxhard all faces have one colour.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `4` |
| medium | 448x448 | `1,4,6` |
| hard | 588x588 | `3,8` |
| xhard | 588x588 | `1,2,7,9,10` |
| xxhard | 588x588 | `3,6,7,9,12` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#cube_unfold).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks cube_unfold --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks cube_unfold --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
