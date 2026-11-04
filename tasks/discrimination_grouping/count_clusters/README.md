# count_clusters

Count the objects that a question selects in a cluttered scene: all objects, those with an attribute, those in a region, overlapping ones, or one icon category drawn in several styles.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> The image has a 3 by 3 grid with sections numbered left to right, top to bottom. How many straight or curved lines have their center in section 1? For a curved line, use its path midpoint. Count each line once.

Answer: `6`

## Tiers

The scene grows from 8-12 objects on easy to 39-48 (or 52-64) on xxhard; every tier draws from the same five question pools.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `3` |
| medium | 448x448 | `20` |
| hard | 448x448 | `6` |
| xhard | 448x448 | `12` |
| xxhard | 448x448 | `15` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#count_clusters).

## Assets

Fluent and mixed-material items need the Fluent Emoji pack ([ASSETS.md](../../../docs/ASSETS.md#fluent-emoji-for-count_clusters)); without it those items are refused, never drawn with substitute pictures. Other items use the bundled icons and fonts.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks count_clusters --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks count_clusters --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
