# Bundled DejaVu fonts

The repository bundles `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` in
`assets/fonts/` for deterministic rendering; every generator loads them from
here (`shared/drawing/fonts.py`). Their content hashes, rather than the
filenames alone, pin the exact font bytes:

| File | SHA-256 |
|---|---|
| `DejaVuSans.ttf` | `3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22` |
| `DejaVuSans-Bold.ttf` | `b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167` |

References:

- Project: <https://dejavu-fonts.github.io/>
- Source repository: <https://github.com/dejavu-fonts/dejavu-fonts>
- Official license page: <https://dejavu-fonts.github.io/License.html>

The complete license is retained at [`LICENSE-DejaVu.txt`](LICENSE-DejaVu.txt).
