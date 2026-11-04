# Assets

## Included

The pinned SVG icon tree, `shared/iconlib/svg_manifest.json` (which also records the
train/test icon split) and DejaVu fonts are included. Do not rebuild or reshuffle the icon manifest: its
order and split membership affect the generated samples. Attribution and
license texts are listed in the repository's [NOTICE](../NOTICE).

Generators load their fonts only from `assets/fonts/`; a missing font is an
error, never a silent fallback to another font.

## find_the_shadow tables

From the VisRO repository root, with the environment active:

```sh
export VISRO_FTS_CACHE=~/.cache/visro/find_the_shadow
VISRO_FTS_WORKERS=6 python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_cache
python -m tasks.discrimination_grouping.find_the_shadow.generator.build.prepare_law --cache "$VISRO_FTS_CACHE"
```

The first command builds the base pool, gate tables and strike census from the
included icons: about 20 minutes on 6 processes, most of it the strike census.
The second installs the fitted law, per-orientation census and pre-screen
bundled under `assets/find_the_shadow_law/`, and builds the orientation table
from the same icons if needed. The finished cache is about 200 MB. These fitted
inputs must not be refitted for a reproduction. The generator validates the
table fingerprints when it loads them; do not enable cache-drift overrides.

## Fluent Emoji for count_clusters

The 1,864 PNGs are generated assets, not part of the source tree. Use the
pinned fluentui-emoji commit and resvg-py version. From the VisRO repository root:

```sh
python -m pip install -r shared/iconlib/requirements_fluent.txt
git clone --filter=blob:none --no-checkout https://github.com/microsoft/fluentui-emoji.git tmp/fluent-source
git -C tmp/fluent-source sparse-checkout init --no-cone
git -C tmp/fluent-source sparse-checkout set --no-cone '/LICENSE' '/README.md' '/assets/*/metadata.json'
git -C tmp/fluent-source checkout 1ffb34c752ecf5d402f04cfb4b392c77f57c54bc
python -m shared.iconlib.prepare_fluent --source-checkout tmp/fluent-source --sparse-patterns tmp/fluent-patterns.txt
git -C tmp/fluent-source sparse-checkout set --no-cone --stdin < tmp/fluent-patterns.txt
python -m shared.iconlib.prepare_fluent --source-checkout tmp/fluent-source --asset-root assets/fluent_countable --manifest tmp/fluent-built-manifest.json --catalog tmp/fluent-built-catalog.json
```

The checked-in `shared/iconlib/fluent_manifest.json` is authoritative; the build
writes temporary manifests so it cannot replace the pinned metadata. The
generator verifies every required PNG against the checked-in hashes before a
Fluent or mixed-material count_clusters item, and refuses to build it otherwise.
The rasteriser and its system libraries must match for byte parity.

**Known gap:** two of the 1,864 pinned hashes, `rgba/1f3f7_fe0f/color.png`
(Label) and `rgba/1f3a9/color.png` (Top hat), are not reproduced by the build
above on every host. Where they differ, the Fluent and mixed-material
count_clusters items are refused rather than generated; they need a matching
prepared pack or a renderer environment that produces all pinned hashes. Do not
replace the pinned manifest or disable the check.

## Tokenizer model

Set `VISRO_TOKENIZER_DIR` to a directory containing `paligemma_tokenizer.model`,
obtained from its publisher under the applicable terms. With it,
overlapping_glyphs applies its 16-token answer filter with that model (the file
must match the pinned SHA-256); without it the glyph runner uses its unfiltered
mode and records that choice. The model is not bundled or downloaded
automatically.

## Environment variables

None is required to generate the tasks that need no prepared assets.

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `VISRO_FTS_CACHE` | find_the_shadow | `~/.cache/visro/find_the_shadow` | cache directory of the shadow tables (required by `build_cache`) |
| `VISRO_FTS_WORKERS` | find_the_shadow cache builders | `6` | worker processes; does not change the tables |
| `VISRO_FTS_TILES` | `build_cache retab` | `92` | comma-separated tile sizes to tabulate; only 92 is used |
| `VISRO_FTS_LAW` | find_the_shadow | `$VISRO_FTS_CACHE/law` | directory of the installed law tables |
| `VISRO_FTS_ALLOW_CACHE_DRIFT` | find_the_shadow | unset | `1` accepts tables whose fingerprints differ; never set it for a reproduction |
| `VISRO_ROTATION_POOLS` | rotation_patterns pool builders | `~/.cache/visro/rotation_patterns` | where rebuilt word pools are written; the bundled pools are used first |
| `VISRO_ROTATION_WORKERS` | rotation_patterns pool builders | `1` | worker processes; does not change the pools |
| `VISRO_TOKENIZER_DIR` | overlapping_glyphs | unset | directory containing `paligemma_tokenizer.model` (see above) |
