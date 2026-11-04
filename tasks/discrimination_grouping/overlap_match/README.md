# overlap_match

Which of four candidate outlines A-D appears, at the same size and orientation, in a stack of overlapping outlines?

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is a letter (A-D), scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Exactly one of the shapes A-D on the right appears in the overlapping outlines on the left, at the same size and orientation (moved only, never rotated or resized). Which one? Answer with the letter.

Answer: `C`

## Tiers

More shapes in the stack, more same-base decoys, and less of the target's outline left visible.

| tier | image (px) | example answer |
|---|---|---|
| easy | 896x504 | `D` |
| medium | 896x504 | `B` |
| hard | 896x504 | `C` |
| xhard | 896x504 | `D` |
| xxhard | 896x504 | `A` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#overlap_match).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks overlap_match --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks overlap_match --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
