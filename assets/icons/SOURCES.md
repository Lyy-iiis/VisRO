# Icon sources

The generators draw icons from three open icon sets, bundled here as the
unmodified upstream SVG files. `shared/iconlib/svg_manifest.json` pins the 16,122
icons in use (id, set, style, path and train/test membership); the generators
read the icons through that manifest, so its order and split membership are
part of the generated data.

| Folder | Upstream | Version | Icons | License | Attribution |
|---|---|---|---:|---|---|
| `mdi/svg/` | [Templarian/MaterialDesign-SVG](https://github.com/Templarian/MaterialDesign-SVG) | 7.4.47 | 7,447 | Pictogrammers Free License; icons under Apache 2.0 ([`mdi/LICENSE`](mdi/LICENSE)) | Material Design Icons by Pictogrammers |
| `gameicons/<author>/` | [game-icons/icons](https://github.com/game-icons/icons) | repository snapshot (the `badges/` folder is not used) | 4,180 | CC BY 3.0 ([`gameicons/LICENSE`](gameicons/LICENSE)) | game-icons.net and each icon's author; the author is the folder directly below `gameicons/` |
| `openmoji/color/` | [hfg-gmuend/openmoji](https://github.com/hfg-gmuend/openmoji), `openmoji-svg-color.zip` | release 17.0.0 | 4,495 | CC BY-SA 4.0 ([`openmoji/LICENSE`](openmoji/LICENSE)) | OpenMoji, the open-source emoji and icon project |

The Fluent Emoji PNGs used by count_clusters are not bundled; they are
prepared from a pinned upstream commit as described in
[docs/ASSETS.md](../../docs/ASSETS.md). The repository's
[NOTICE](../../NOTICE) lists every bundled third-party asset.
