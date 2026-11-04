# metro_map

Stops, transfers, station counts and nearest-station questions on a metro map.

Category: `paths_connectivity` (Paths and connectivity). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> In this subway map, what is the minimum number of stops between "Las" station and "Sut" station? (Transfers are allowed, excluding the starting and ending stations)

Answer: `10`

## Tiers

3-4 lines and 14-30 stations on easy, 5-6 lines and 38-50 stations from hard on; xhard and xxhard add competing routes of different lengths, so that finding the minimum takes work. xxhard uses held-out map styles.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `4` |
| medium | 448x448 | `8` |
| hard | 448x448 | `10` |
| xhard | 448x448 | `11` |
| xxhard | 448x448 | `8` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#metro_map).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks metro_map --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks metro_map --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
