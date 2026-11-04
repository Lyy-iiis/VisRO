# overlapping_glyphs

Read overlapping, differently coloured letters and digits in the order of the colour boxes.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> The image contains 5 uppercase letters or Arabic numerals in different colors. Read the characters matching the colored boxes on the top, in box order from left to right. Separate the characters with commas.

Answer: `R,5,K,C,7`

## Tiers

Three glyphs on easy to six on xhard, split into more groups; xxhard has 5-8 glyphs with out-of-distribution materials.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `K,N,W` |
| medium | 448x448 | `A,Z,X,Y` |
| hard | 448x448 | `R,5,K,C,7` |
| xhard | 448x448 | `K,8,L,O` |
| xxhard | 448x448 | `4,R,K,6` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#overlapping_glyphs).

## Assets

Bundled fonts. Optionally a tokenizer model enables a 16-token answer filter ([ASSETS.md](../../../docs/ASSETS.md#tokenizer-model)); the record states which mode was used.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks overlapping_glyphs --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks overlapping_glyphs --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
