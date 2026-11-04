#!/usr/bin/env python3
"""Prepare a pinned, curated Fluent emoji pack as original-colour RGBA PNGs.

This is an offline build from a sparse checkout. It does not change the shared
IconLibrary manifest or the count_clusters question distribution. SVG assets
use resvg-py so gradients and SVG filters are retained.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage

from shared.iconlib.fluent_splits import apply_plan, split_summary, write_selected_manifest
from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text

SOURCE_REPO = "https://github.com/microsoft/fluentui-emoji"
SOURCE_COMMIT = "1ffb34c752ecf5d402f04cfb4b392c77f57c54bc"
STYLE_DIRS = {"3D": "3d", "Color": "color", "Flat": "flat", "High Contrast": "high_contrast"}
STYLES = tuple(STYLE_DIRS.values())


# schema label of the pinned fluent_manifest.json (see fluent.SCHEMA_VERSION)
_SCHEMA_VERSION = seed_text(164588486048695051706877160126359422275682418761603546363033137)

def git(checkout: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(checkout), *args])


def source_index(checkout: Path) -> dict:
    result = {}
    # Do not request -l (blob sizes): a partial clone would need to fetch
    # unrelated blobs merely to report their sizes.
    for raw in git(checkout, "ls-tree", "-rz", SOURCE_COMMIT, "assets").split(b"\0"):
        if not raw:
            continue
        info, path = raw.decode().split("\t", 1)
        mode, kind, sha = info.split()
        if kind == "blob":
            result[path] = {"git_blob_sha1": sha}
    return result


def variants_for(name: str, index: dict) -> list[dict]:
    prefix = f"assets/{name}/"
    variants = []
    for path, details in index.items():
        if not path.startswith(prefix):
            continue
        parts = Path(path).parts
        if len(parts) < 4 or parts[-2] not in STYLE_DIRS:
            continue
        if Path(path).suffix.lower() not in (".png", ".svg"):
            continue
        # Extended body-part entries use only the fluentui-emoji Default branch.
        # Skin tones are excluded here and are never interpreted as poses.
        if len(parts) == 5 and parts[2] != "Default":
            continue
        if len(parts) not in (4, 5):
            raise ValueError(f"curated identity needs explicit variant mapping: {path}")
        variants.append({"style": STYLE_DIRS[parts[-2]], "source_path": path, **details})
    found = [v["style"] for v in variants]
    if sorted(found) != sorted(STYLES):
        raise ValueError(f"{name}: expected exactly one asset per style; got {found}")
    return sorted(variants, key=lambda v: STYLES.index(v["style"]))


def source_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def image_metrics(image: Image.Image) -> dict:
    rgba = np.asarray(image)
    mask = rgba[..., 3] > 40
    if not mask.any() or mask.all():
        raise ValueError("asset must have both foreground and transparent background")
    ys, xs = np.where(mask)
    labels, count = ndimage.label(mask)
    areas = np.bincount(labels.ravel())[1:]
    significant = int((areas >= max(8, int(mask.sum() * .01))).sum())
    return {
        "foreground_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "foreground_centroid": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
        "foreground_pixels": int(mask.sum()),
        "foreground_fraction": round(float(mask.mean()), 6),
        "alpha_threshold": 40,
        "component_count": int(count),
        "significant_component_count": significant,
        "largest_component_fraction": round(float(areas.max() / mask.sum()), 6),
        "alpha_range": [int(rgba[..., 3].min()), int(rgba[..., 3].max())],
    }


def build_variant(job: tuple) -> dict:
    checkout, output, unicode_id, variant, size = job
    source = checkout / variant["source_path"]
    data = source.read_bytes()
    source_bytes = len(data)
    if source_blob_sha(data) != variant["git_blob_sha1"]:
        raise ValueError(f"source differs from pinned Git blob: {source}")
    if source.suffix.lower() == ".svg":
        import resvg_py
        data = resvg_py.svg_to_bytes(svg_string=data.decode(), width=size, height=size,
                                    skip_system_fonts=True)
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGBA")
    if image.size != (size, size):
        # Source viewboxes are square in this pinned pack. Do not stretch an
        # unexpected rectangular source while normalizing output dimensions.
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        square = Image.new("RGBA", (size, size))
        square.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
        image = square
    metrics = image_metrics(image)
    relative = Path("rgba") / unicode_id / f"{variant['style']}.png"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    result = {
        **variant, "source_bytes": source_bytes, "path": relative.as_posix(), "format": "png",
        "width": image.width, "height": image.height,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size, **metrics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curation", type=Path, default=(_ROOT / 'shared/iconlib/fluent_curation.json'))
    parser.add_argument("--asset-root", type=Path, default=(_ROOT / 'assets/fluent_countable'))
    parser.add_argument("--manifest", type=Path, default=(_ROOT / 'shared/iconlib/fluent_manifest.json'))
    parser.add_argument("--catalog", type=Path, default=(_ROOT / 'shared/iconlib/fluent_catalog.json'))
    parser.add_argument("--split-plan", type=Path, default=(_ROOT / 'shared/iconlib/fluent_splits.json'))
    parser.add_argument("--catalog-only", action="store_true",
                        help="prepare an inventory without an active experiment manifest")
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--sparse-patterns", type=Path,
                        help="write sparse-checkout patterns and exit before rendering")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.size < 32 or args.workers < 1:
        parser.error("size must be >=32 and workers >=1")
    if args.catalog.resolve() == args.manifest.resolve():
        parser.error("catalog and selected manifest must use different paths")
    checkout, output = args.source_checkout.resolve(), args.asset_root.resolve()
    if git(checkout, "rev-parse", "HEAD").decode().strip() != SOURCE_COMMIT:
        raise ValueError(f"source checkout must be pinned to {SOURCE_COMMIT}")
    curation = json.loads(args.curation.read_text())
    if curation["source_commit"] != SOURCE_COMMIT:
        raise ValueError("curation source commit mismatch")
    allowed = {"core", "extended"} if args.include_extended else {"core"}
    curated = [e for e in curation["entries"] if e["status"] in allowed]
    if not curated:
        raise ValueError("curation selected no entries")
    if len({e["source_name"] for e in curated}) != len(curated):
        raise ValueError("duplicate curated source identity")
    index = source_index(checkout)
    entries = []
    for entry in sorted(curated, key=lambda e: e["source_name"]):
        metadata = json.loads((checkout / "assets" / entry["source_name"] / "metadata.json").read_text())
        unicode_id = metadata["unicode"].lower().replace(" ", "_")
        entries.append({
            "identity_id": f"fluent/{unicode_id}", "source_name": entry["source_name"],
            "unicode": unicode_id, "class_id": entry["class_id"],
            "noun": entry["noun"], "plural": entry["plural"], "group": metadata["group"],
            "pool": entry["status"], "split_group": entry["class_id"], "split": "catalog",
            "pose_id": None, "countable_unit": True,
            "semantic_review": "curated_identity_names; see curation reasons",
            "prompt_mode": entry.get("prompt_mode", "reference"),
            "visual_qa": entry.get("visual_qa", "pending"),
            "curation_reason": entry["reason"],
            "counting_notes": entry.get("counting_notes", []),
            "family_ids": entry.get("family_ids", []),
            "counting_contract": "Count primary targets of the requested class; not every depicted part or accessory",
            "variants": variants_for(entry["source_name"], index),
        })
    if args.sparse_patterns:
        patterns = ["/LICENSE", "/README.md", "/assets/*/metadata.json"]
        patterns.extend(f"/{v['source_path']}" for e in entries for v in e["variants"])
        args.sparse_patterns.parent.mkdir(parents=True, exist_ok=True)
        args.sparse_patterns.write_text("\n".join(patterns) + "\n")
        print(json.dumps({"identities": len(entries), "classes": len({e['class_id'] for e in entries}),
                          "assets": sum(len(e["variants"]) for e in entries),
                          "patterns": str(args.sparse_patterns)}, indent=2))
        return
    missing = [v["source_path"] for e in entries for v in e["variants"] if not (checkout / v["source_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} assets missing; first: {missing[0]}; update sparse checkout first")
    source = {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT, "license": "MIT",
              "attribution": "Fluent Emoji, Copyright (c) Microsoft Corporation"}
    plan = None
    if not args.catalog_only:
        if not args.split_plan.is_file():
            raise FileNotFoundError("split plan missing; prepare --catalog-only, then run iconlib.fluent_splits")
        plan = json.loads(args.split_plan.read_text())
        # Validate the fixed class membership before rendering any PNGs.
        apply_plan({"source": source, "entries": entries}, plan)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(checkout, output, e["unicode"], v, args.size) for e in entries for v in e["variants"]]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(build_variant, jobs), 1):
            results.append(result)
            if n % 200 == 0:
                print(f"rendered {n}/{len(jobs)}", flush=True)
    position = 0
    for entry in entries:
        count = len(entry["variants"])
        entry["variants"] = results[position:position + count]
        position += count
    manifest_path = args.manifest.resolve()
    catalog_path = args.catalog.resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    import resvg_py
    manifest = {"schema_version": _SCHEMA_VERSION, "source": source,
                "asset_root": os.path.relpath(output, catalog_path.parent),
                "styles": list(STYLES), "rasterizer": f"resvg-py=={resvg_py.__version__}",
                "curation_sha256": hashlib.sha256(args.curation.read_bytes()).hexdigest(),
                "scope": "catalog",
                "split_policy": "Unpartitioned full asset inventory. Use fluent_manifest.json for the selected experiment split.",
                "pose_policy": "Fixed source view; no 3D models or arbitrary true pose augmentation",
                "skin_tone_policy": "Direct assets or Default branch only; no skin-tone augmentation",
                "family_constraints": curation.get("family_constraints", []),
                "entries": entries}
    catalog_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    selected_summary = None
    if plan is not None:
        selected = write_selected_manifest(manifest, plan, manifest_path, output)
        selected_summary = split_summary(selected)
        (output / "split_summary.json").write_text(json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n")
    (output / "LICENSE").write_bytes((checkout / "LICENSE").read_bytes())
    (output / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    group_stats = {}
    for group in sorted({e["group"] for e in entries}):
        grouped = [e for e in entries if e["group"] == group]
        group_stats[group] = {"identities": len(grouped), "classes": len({e["class_id"] for e in grouped})}
    summary = {
        "source": source, "catalog": str(catalog_path),
        "manifest": str(manifest_path) if plan is not None else None,
        "classes": len({e["class_id"] for e in entries}), "identities": len(entries), "assets": len(results),
        "styles": dict(collections.Counter(v["style"] for v in results)),
        "png_bytes": sum(v["bytes"] for v in results),
        "unique_png_hashes": len({v["sha256"] for v in results}),
        "groups": group_stats, "active_selection": selected_summary,
        "multiple_significant_component_assets": sum(v["significant_component_count"] > 1 for v in results),
        "readability_note": "Alpha/component metrics are diagnostics, not proof of semantic countability or shading-free masks.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
