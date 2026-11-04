"""Rebuild find_the_different_render_alias.json, the render-alias table the adapter merges into artwork classes.

Run from visro/:
    python -m tasks.discrimination_grouping.find_the_different.generator.build.render_alias
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

import shared.iconlib as iconlib
from ...adapter import IconClasses, RENDER, RENDER_ALIAS_PATH, RENDER_SIZES, library_digest, svg_classes

_ALIAS_LIB = None


def _alias_render_hashes(i: int) -> list:
    """sha256 of the RGBA bytes of icon i's live render at each RENDER_SIZES (None if fully transparent)."""
    e = _ALIAS_LIB.icons[i]
    out = []
    for size in RENDER_SIZES:
        img = _ALIAS_LIB.render(e, size, (30, 30, 30) if e["style"] == "silhouette" else None)
        a = np.asarray(img.convert("RGBA"))
        out.append(hashlib.sha256(a.tobytes()).hexdigest() if a[..., 3].any() else None)
    return out


def write_render_alias_table(path=RENDER_ALIAS_PATH, workers: int = 6) -> dict:
    """Recompute the render-alias table (about 15 CPU-minutes) and write it to `path`.

    Renders every manifest icon live (the release render path, silhouettes tinted (30,30,30); a tint only
    recolours opaque pixels, so equality holds for every tint) at every production size, and keeps the
    classes of ids whose RGBA renders are byte-identical at one or more sizes and whose SVGs fall in more
    than one byte/normalised class.
    """
    global _ALIAS_LIB
    _ALIAS_LIB = iconlib.IconLibrary(default_split="train")
    ids, sha, art = svg_classes(_ALIAS_LIB)
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        hashes = dict(zip(ids, pool.map(_alias_render_hashes, range(len(ids)), chunksize=64)))
    same = IconClasses(ids)
    per_size = {}
    for k, size in enumerate(RENDER_SIZES):
        by = {}
        for i in ids:
            if hashes[i][k] is not None:
                by.setdefault(hashes[i][k], []).append(i)
        per_size[size] = sorted(sorted(g) for g in by.values() if len(g) > 1)
        same.merge_by(ids, lambda i, k=k: hashes[i][k])
    classes = {}
    for i in ids:
        classes.setdefault(same.find(i), []).append(i)
    identical = [sorted(g) for g in classes.values() if len(g) > 1]
    groups = sorted(g for g in identical if len({art.find(i) for i in g}) > 1)
    first = per_size[RENDER_SIZES[0]]
    table = {
        "about": "find_the_different render aliases: ids whose SVGs differ but whose live renders are "
                 "pixel-identical; merged into one artwork class by icon_index()",
        "library_digest": library_digest(_ALIAS_LIB.icons, sha),
        "render": {"path": RENDER["path"], "cairosvg": RENDER["cairosvg"], "cairo": RENDER["cairo"],
                   "sizes": list(RENDER_SIZES), "silhouette_tint": [30, 30, 30],
                   "key": "sha256 of the RGBA bytes of the render"},
        "stats": {
            "icons": len(ids),
            "fully_transparent_renders": sum(all(h is None for h in hashes[i]) for i in ids),
            "render_identical_groups": len(identical),
            "ids_in_render_identical_groups": sum(map(len, identical)),
            "per_size_partitions_equal": all(per_size[s] == first for s in RENDER_SIZES),
            "groups_spanning_svg_classes": len(groups),
            "ids_in_them": sum(map(len, groups)),
        },
        "groups": groups,
    }
    _write_alias_table(path, table)
    return table


def _write_alias_table(path, table: dict) -> None:
    """JSON with one alias group per line."""
    head = json.dumps({k: v for k, v in table.items() if k != "groups"}, indent=1)[:-2]
    groups = ",\n".join("  " + json.dumps(g) for g in table["groups"])
    text = f'{head},\n "groups": [\n{groups}\n ]\n}}\n'
    assert json.loads(text) == table
    Path(path).write_text(text)


if __name__ == "__main__":
    write_render_alias_table()
