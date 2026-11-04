# find_the_same

Among numbered pictures that include near-twins and singles, list every pair of identical pictures.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> The image shows 16 numbered pictures. List every pair of identical pictures (identical means the same shape, parts, colour and orientation, so rotated, mirrored or edited copies differ; identical items come only in pairs). Answer each pair as a-b, separated by commas, in any order.

Answer: `1-12,2-9,3-11,5-16,6-13`

## Tiers

More pictures (10 on easy to 25 on xxhard) and more pairs to find; near-twins differ by rotation, mirroring or a small edit.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `1-9,6-8,7-10` |
| medium | 448x448 | `1-3,2-7,9-12` |
| hard | 448x448 | `1-12,2-9,3-11,5-16,6-13` |
| xhard | 588x588 | `1-4,3-14,5-15,6-10,7-19,18-20` |
| xxhard | 588x588 | `1-7,2-9,3-13,4-20,5-14,8-23,10-25,12-19` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#find_the_same).

train and val draw from the icon library's train icons; eval and ood from the held-out test icons.

## Assets

Only the bundled icon tree and fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks find_the_same --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks find_the_same --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
