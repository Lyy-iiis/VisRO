# count_points_on_lines

Blue segments are drawn over a grid of dots; give each segment's dot count and the number of distinct dots covered.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Blue segments are drawn over a grid of black dots; a segment passes through a dot only if it crosses its centre. Answer each segment's dot count in descending order, comma-separated, then ';' and the number of distinct dots covered, no spaces, e.g. 2,2;3.

Answer: `8,7,6,6,6,6,5,5,4,4,4,4;58`

## Tiers

More segments over a denser dot grid (448 px canvas on easy and medium, 672 px from hard on). Slopes 0, infinity and +-1 never occur, every image has forced near misses, and the whole answer must be right.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `6,5,4;14` |
| medium | 448x448 | `6,5,5,4;20` |
| hard | 672x672 | `8,7,6,6,6,6,5,5,4,4,4,4;58` |
| xhard | 672x672 | `10,7,7,6,6,6,6,6,5,5,4;63` |
| xxhard | 672x672 | `9,8,7,7,7,7,7,6,6,5,5,5,4;74` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#count_points_on_lines).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks count_points_on_lines --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks count_points_on_lines --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
