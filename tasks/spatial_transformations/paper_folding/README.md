# paper_folding

Fold and punch a grid sheet in stages, unfold it, and report which queried cells have holes.

Category: `spatial_transformations` (Spatial transformations and structure). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Fold the 8x8 grid sheet in order without turning it, moving the named half (crease x/y = grid line, 0 at top left): x4 left, punch P1; y4 top; upper right triangle, punch P2; then unfold. Each panel shows the packet in the original frame with red holes punched through all layers (earlier holes omitted). Give one 10-digit string: per cell (row,column) below in order, 1 if a hole is centred there, else 0: (8,7),(5,5),(3,1),(7,3),(2,1),(5,4),(8,4),(6,3),(6,6),(6,2).

Answer: `1000101110`

## Tiers

More folds and punches, including diagonal folds, on 8x8 and then 16x16 sheets, with more cells to report.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `110001` |
| medium | 448x448 | `00101101` |
| hard | 448x448 | `1000101110` |
| xhard | 448x448 | `100101010011` |
| xxhard | 448x448 | `0101001000011111` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#paper_folding).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks paper_folding --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks paper_folding --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
