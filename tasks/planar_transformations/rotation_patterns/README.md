# rotation_patterns

Identify which figures are a coded base figure turned clockwise by given multiples of 22.5 degrees.

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Figures 1-12 are the bold-framed base (far left) turned clockwise by different multiples of 22.5 degrees; the bead lengths on the 16 spokes are the code. Which figure is the base turned clockwise by each angle: 292.5, 90, 67.5, 270, 157.5, 247.5, 45, 135, 202.5? Answer the figure numbers in order, comma-separated.

Answer: `8,7,5,12,10,4,9,11,1`

## Tiers

More figures and angles; xhard adds angles that no figure matches, and xxhard adds rotations relative to a named figure.

| tier | image (px) | example answer |
|---|---|---|
| easy | 1232x224 | `5,6` |
| medium | 1232x224 | `1,6,3` |
| hard | 1232x364 | `8,7,5,12,10,4,9,11,1` |
| xhard | 1232x364 | `3,1,none,8,6,none,12,10,7,5,2,9` |
| xxhard | 1232x364 | `7,3,9,10,none,2,11,5,6,1,none,8` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#rotation_patterns).

## Assets

The word pools ship with the task (`generator/pools/`); nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks rotation_patterns --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks rotation_patterns --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
