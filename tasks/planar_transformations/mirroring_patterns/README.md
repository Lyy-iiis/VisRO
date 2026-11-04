# mirroring_patterns

Apply reflections restricted to rectangles of a colour grid, then read the colours of the asked cells.

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Cells are (row,column), numbered 1–8 as labelled. Apply each step in order to the previous result: reverse the order of the given rows or columns inside the rows×columns rectangle.
> 3–7×2–5 rows
> 3–5×4–5 columns
> Then give the colors of (5,5),(3,4),(4,5),(5,4) in order as 4 letters, no separators: R red, B blue, G green, Y yellow, P purple, W white.

Answer: `BRRW`

## Tiers

Larger grids (4x4 to 12x12), more reflection steps and more cells to read.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `YGB` |
| medium | 448x448 | `BYG` |
| hard | 448x448 | `BRRW` |
| xhard | 448x448 | `GYWGWPYP` |
| xxhard | 448x448 | `GPPBBYRRRRBP` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#mirroring_patterns).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks mirroring_patterns --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks mirroring_patterns --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
