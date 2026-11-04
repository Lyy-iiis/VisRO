# find_the_shadow

Match each coloured pattern to its one dark shadow among look-alike silhouettes and list the shadows that match nothing.

Category: `discrimination_grouping` (Discrimination and grouping). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> Each colored pattern A-F has exactly one same-shape dark shadow among 1-22, none shared, so 16 match nothing. Shadows are never rotated, mirrored or rescaled; holes count. Look-alikes differ by one small edge edit; the match differs nowhere. Answer pairs in letter order, then ';unmatched:' and the rest ascending, e.g. Y-93,Z-91;unmatched:92,94.

Answer: `A-15,B-22,C-7,D-9,E-20,F-12;unmatched:1,2,3,4,5,6,8,10,11,13,14,16,17,18,19,21`

## Tiers

More shadows and more look-alike clones per pattern; a clone differs from its pattern by one small edge edit, while the true shadow differs nowhere.

| tier | image (px) | example answer |
|---|---|---|
| easy | 644x364 | `A-3,B-4,C-8,D-9,E-5;unmatched:1,2,6,7,10,11,12,13` |
| medium | 728x364 | `A-5,B-14,C-3,D-9,E-6,F-8;unmatched:1,2,4,7,10,11,12,13,15` |
| hard | 728x476 | `A-15,B-22,C-7,D-9,E-20,F-12;unmatched:1,2,3,4,5,6,8,10,11,13,14,16,17,18,19,21` |
| xhard | 728x616 | `A-5,B-21,C-2,D-24,E-16,F-17;unmatched:1,3,4,6,7,8,9,10,11,12,13,14,15,18,19,20,22,23,25,26,27,28,29` |
| xxhard | 728x616 | `A-23,B-3,C-21,D-7,E-27;unmatched:1,2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,25,26,28,29,30` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#find_the_shadow).

No split holds out icons: every split draws from the icon library's train icons, and val, eval and ood are kept disjoint from train by base-icon set and by image.

## Assets

Needs the find_the_shadow tables, built once from the bundled icons ([ASSETS.md](../../../docs/ASSETS.md#find_the_shadow-tables)).

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks find_the_shadow --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks find_the_shadow --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
