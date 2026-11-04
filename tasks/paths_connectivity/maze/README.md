# maze

Reachability, distance-order and radius questions on a rendered maze.

Category: `paths_connectivity` (Paths and connectivity). Each item is one image and one question; the answer is a letter (A-B), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> By corridor distance, which start is closer to the unlabeled target marker: A or B? Answer A or B.

Answer: `A`

## Tiers

Larger mazes with more junctions inside the region the answer depends on; rectangular, hexagonal and circular lattices; the question forms are scheduled so that no straight-line or component-size cue predicts the answer.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `A` |
| medium | 448x448 | `A` |
| hard | 448x448 | `A` |
| xhard | 448x448 | `A` |
| xxhard | 448x448 | `A` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#maze).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks maze --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks maze --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
