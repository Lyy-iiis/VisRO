# path_tracing

Follow tangled curves from a labelled endpoint to the label at the other end.

Category: `paths_connectivity` (Paths and connectivity). Each item is one image and one question; the answer is free-form text in the format the question states, scored as described in [DATASET.md](../../../docs/DATASET.md#scoring).

## Example (hard)

> What is 6 connected to? Answer with only the marker label.

Answer: `B`

## Tiers

More curves and crossings; some questions ask about two endpoints at once. xxhard uses a separate out-of-distribution generator.

| tier | image (px) | example answer |
|---|---|---|
| easy | 448x448 | `1` |
| medium | 448x448 | `23` |
| hard | 448x448 | `B` |
| xhard | 448x448 | `7` |
| xxhard | 448x448 | `65` |

easy to xhard make up `train` and `eval`; xxhard is the out-of-distribution tier (`ood`). The exact parameters of each tier and the verifier's checks are in [TASKS.md](../../../docs/TASKS.md#path_tracing).

## Assets

Only the bundled fonts; nothing to prepare.

## Files

- `adapter.py`: tier definitions; builds one seeded item.
- `verifier.py`: independent answer correctness and uniqueness checks.
- `generator/`: task-specific sampling, geometry and rendering.
- Libraries shared with other tasks live under `shared/` in the `visro/` directory.

From the `visro/` directory:

```sh
python generate.py generate --tasks path_tracing --tiers easy --n 2 --seed 20261020 --out out/example
python generate.py verify --tasks path_tracing --tiers easy --root out/example --rebuild 2
```

See also the [record format](../../../docs/DATASET.md) and the [reproduction notes](../../../REPRODUCING.md).
