# tangram_reconstruction

Cover a gray triangle-grid silhouette with the numbered pieces, giving each piece's position (and rotation).

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Cover the gray target with every numbered orange piece once, no gaps or overlaps, leaving blue triangles uncovered; count grid squares (4 triangles each), not pixels. Pieces may turn but not flip. Answer a JSON array without spaces, one [x,y,k] per piece in number order: turn it k clockwise quarter-turns (0-3, smallest k if equal), then put the top-left of its grid-aligned bounding box (corner triangles included) at grid point (x,y), x right, y down.

Answer: `[[0,1,3],[2,1,1],[1,0,1]]`

## Tiers

More pieces; from medium on the pieces may be turned.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `[[0,2,0],[1,0,0]]` |
| medium | 448x448 | `[[2,1,0],[0,1,2],[1,0,0]]` |
| hard | 448x448 | `[[0,1,3],[2,1,1],[1,0,1]]` |
| xhard | 448x448 | `[[1,0,1],[3,1,3],[1,1,2],[2,5,0],[0,1,1],[3,0,0]]` |
| xxhard | 448x448 | `[[3,1,3],[2,0,3],[0,0,0],[4,0,0],[0,1,1],[2,2,1],[1,0,0],[5,1,1]]` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#tangram_reconstruction).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks tangram_reconstruction --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks tangram_reconstruction --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
