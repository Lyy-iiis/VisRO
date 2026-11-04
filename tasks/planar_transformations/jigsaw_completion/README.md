# jigsaw_completion

Fill the holes of an edge-strip board with lettered pieces, giving each piece's rotation.

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Fill each numbered gray hole with a lettered piece so colored strips continue across every shared edge, including between holes; board-border sides are free. Pieces turn clockwise in quarter turns, never flip, and are used at most once. Answer letter+turns (0-3, from as drawn) per hole in hole order, comma-separated, e.g. X1,Y0,Z3.

Answer: `C3,E2`

## Tiers

More holes and more decoy pieces: decoys first differ by a recoloured strip, from hard on by a strip shifted one position.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `B3,D1` |
| medium | 448x448 | `A3,E0` |
| hard | 448x448 | `C3,E2` |
| xhard | 588x588 | `F1,B1,G0` |
| xxhard | 588x588 | `F2,A1,B1,D1,J3` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#jigsaw_completion).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks jigsaw_completion --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks jigsaw_completion --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
