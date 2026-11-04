# flow_free

Questions about the pipes of a solved Flow Free board: the length or the number of turns of one pipe or of every pipe.

Category: `paths_connectivity` (Paths and connectivity). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> All pipe colours in this image are unique. Follow the red pipe. A turn is an internal square where the pipe enters in one direction and leaves in a perpendicular direction. Straight-through squares count as zero, and endpoint squares never count as turns. How many turns does it make? Answer with a single number.

Answer: `19`

## Tiers

Longer pipes with more turns. xxhard shifts the path program and the pipe appearance (curved pipes, textures, held-out widths and colours) in five out-of-distribution strata.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `24,7,38,12` |
| medium | 448x448 | `36` |
| hard | 448x448 | `19` |
| xhard | 448x448 | `2,9,18,19,26,37,57,60` |
| xxhard | 448x448 | `12,74,113,26` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#flow_free).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks flow_free --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks flow_free --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
