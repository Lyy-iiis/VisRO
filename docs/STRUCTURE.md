# Source structure

The VisRO repository root contains the complete source release:

```text
VisRO/
├── generate.py       # common command-line entrypoint
├── pyproject.toml    # installs the visro package (reader + scorer; generator extra)
├── requirements.txt
├── tasks/            # four categories, 24 tasks
├── visro/            # common generation and verification workflow
├── shared/           # libraries used by several tasks
├── assets/           # required fonts, icons and fitted inputs
├── docs/             # task reference, record format, assets, validation
├── LICENSE
├── NOTICE
├── README.md
└── REPRODUCING.md
```

`LICENSE` (Apache-2.0, for the code) and `NOTICE` (third-party code and the
bundled icons and fonts, with their licenses) are at the repository root.

The four category directories contain the tasks directly, and every task has
the same public interface.

```text
tasks/
├── discrimination_grouping/
│   ├── count_clusters/
│   ├── count_points_on_lines/
│   ├── find_the_same/
│   ├── find_the_different/
│   ├── find_the_shadow/
│   ├── overlapping_glyphs/
│   └── overlap_match/
├── paths_connectivity/
│   ├── flow_free/
│   ├── metro_map/
│   ├── maze/
│   └── path_tracing/
├── planar_transformations/
│   ├── jigsaw_completion/
│   ├── reconstruction/
│   ├── tangram_reconstruction/
│   ├── mirroring_patterns/
│   ├── rotation_patterns/
│   └── logic_matrix/
└── spatial_transformations/
    ├── count_3d_blocks/
    ├── block_delta/
    ├── cube_unfold/
    ├── paper_folding/
    ├── three_d_views/
    ├── view_reconstruction/
    └── viewpoint_rotation/
```

## Inside a task

```text
<task>/
├── README.md       # what the task asks, its tiers and its asset requirements
├── adapter.py      # task/tier definitions; build a seeded, standard Item
├── verifier.py     # independently check the answer and task constraints
├── generator/      # task-specific sampling, geometry and rendering (a package)
└── __init__.py     # Python package marker
```

A task may also keep a small JSON table used by its adapter or verifier.
`generator/` holds what the task's generator actually uses: a `base/`
directory for primitives or several modules for sampling policies. Path
Tracing keeps its in-distribution and out-of-distribution algorithms under
`generator/id/` and `generator/ood/`; both are selected by the same adapter.

block_delta, count_3d_blocks, three_d_views and viewpoint_rotation keep their
xhard/xxhard generator in `generator/xhard.py`.

The generator and verifier have separate responsibilities. Independent answer
checks are kept even when they use similar geometry: sharing the answer
implementation would let the same bug appear in both generation and checking.

## How an adapter loads its generator

Every `generator/` is a regular Python package (it has an `__init__.py`). The
adapter and the verifier import its modules package-qualified, as
`tasks.<category>.<task>.generator.<module>`, and the modules inside
`generator/` import each other with relative imports. No code changes
`sys.path`, loads a module from a file path (`spec_from_file_location`) or
imports a generator module by its bare name; commands run from the repository root, so
`tasks`, `shared` and `visro` are importable as packages.

Offline asset builders, which prepare a pinned pool or cache once rather than
run during generation, live in the task's `generator/build/`. The scripts of
find_the_shadow and find_the_different run from the repository root as
`python -m tasks.<category>.<task>.generator.build.<module>`; rotation_patterns'
pool builders are called by `generator/word_pools.py` when a pinned pool file
is missing.

A module named `<family>_low.py` (`jigsaw_joint_low`, `logic_low`,
`mirrors_low`, `paper_low`) builds a task's lower tiers (easy and medium, plus
hard in paper_folding); its upper tiers come from another module named in the
adapter docstring, or from `shared/puzzles/symbolic.py`.

## Shared code and commands

`visro/` handles the common CLI, seeds, records, generation workers and dataset
verification, and reads and scores the published dataset:

| Module | Role |
|---|---|
| `api.py` | the adapter / verifier contract, `Item`, the public record fields |
| `seeding.py` | per-item seeds and digests |
| `driver.py` | generate one (task, tier) cell: build, verify, de-duplicate, write |
| `checks.py` | re-check finished cells: contract, duplicates, rebuild, verifier |
| `runners.py` | helpers for generators that write a full record per scheduled index |
| `registry.py` | the 24 tasks and their categories, the tiers and the splits |
| `cli.py`, `preview.py` | command line and HTML preview |
| `dataset.py` | read a downloaded dataset: tar shards or extracted shards, optional checksums (stdlib) |
| `scoring.py` | score predictions on the published records; `python -m visro score` (stdlib) |

`shared/` contains code needed across tasks:

| Path | Role |
|---|---|
| `iconlib/` | icon library: the pinned SVG icon manifest (`svg_manifest.json`) and its rendering, the Fluent catalogue and its preparation scripts |
| `drawing/` | bundled-font loading (`fonts.py`), grid geometry types (`geometry.py`; maze, path_tracing), and the isometric cube renderer (`isometric.py`; `placement` fixes a camera from a heightmap and `render_fixed` draws any heightmap with it) |
| `extra_difficulty.py` | helpers for the xhard/xxhard tiers: tier image size, final-resolution checks, heightmap proposals, record helper |
| `puzzles/` | helpers used by more than one of jigsaw_completion, logic_matrix, mirroring_patterns, paper_folding and tangram_reconstruction: `drawing.py` canvas and colours, `symbolic.py` hard-tier builders, `builders.py` record wrapper, `runtime.py` records and per-item RNG, `oracles.py` independent answer checks |

Code that only one task uses lives in that task's `generator/`.

```sh
python generate.py list
python generate.py generate --tasks paths_connectivity --tiers easy --n 1 --out out/paths
```

Asset preparation: `find_the_shadow/generator/build/` (`build_cache`, `prepare_law`) and
`shared/iconlib/prepare_fluent.py`; see [ASSETS.md](ASSETS.md). See
[VALIDATION.md](VALIDATION.md) for checks you can rerun.
