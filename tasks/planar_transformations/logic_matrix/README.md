# logic_matrix

Complete a matrix from attribute rules (multiple choice), or compute the grid produced by a chain of boolean operations shown by demonstrations.

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is a letter on some tiers and free-form text on others, as the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Each label (X and Y) is one tile-wise operation, the same in both its demonstrations (left, label, right = result): OR, AND, XOR or left-minus-right; labels may share one. Start from I1; each step turns the current grid clockwise by its angle, then applies its label with the current grid on the left and its input on the right. Steps: 0 degrees, X, I2; 270 degrees, Y, I3. Answer the final grid as 4 rows of 4 digits (1 filled, 0 empty), top to bottom, separated by /.

Answer: `1000/0100/0001/0100`

## Tiers

easy and medium are attribute matrices with more changing attributes; hard to xxhard chain two or three operations, each shown only by demonstrations, with rotations on 4x4 to 7x7 grids.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `B` |
| medium | 448x448 | `A` |
| hard | 448x448 | `1000/0100/0001/0100` |
| xhard | 588x588 | `00111/01000/00011/01001/10011` |
| xxhard | 588x588 | `1100101/1001111/0110100/0111110/1000010/0111010/1010111` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#logic_matrix).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks logic_matrix --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks logic_matrix --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
