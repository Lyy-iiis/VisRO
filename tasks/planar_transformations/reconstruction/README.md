# reconstruction

Which option's coloured pieces tile the target shape exactly?

Category: `planar_transformations` (Planar transformations and composition). Each item is one image and one question; the answer is a letter (A-D), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Which option's coloured pieces, each used exactly once, tile the top shape exactly with no overlap or leftover cells? Pieces may be moved and turned by 90 degrees but not flipped; options may be drawn at another scale. Answer with the letter.

Answer: `D`

## Tiers

More pieces, and a target that fills more of its bounding box, so that its outline constrains the tiling less.

| tier | image (px) | example answer |
|---|---|---|
| easy | 840x532 | `B` |
| medium | 840x532 | `C` |
| hard | 840x532 | `D` |
| xhard | 840x532 | `D` |
| xxhard | 840x532 | `D` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#reconstruction).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks reconstruction --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks reconstruction --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
