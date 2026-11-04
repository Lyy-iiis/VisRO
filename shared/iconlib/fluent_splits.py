"""Freeze class splits for Fluent's single-identity, four-style subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path

from visro.paths import ROOT as _ROOT
from visro.seeding import seed_text

# schema label stored in the pinned fluent_splits.json
PLAN_SCHEMA = seed_text(38321243144733610494669765294790283697325942240015921)
STYLES = {"3d", "color", "flat", "high_contrast"}


# fixed namespace of the class ranking (changing it changes the split)
_RANK_NAMESPACE = seed_text(136144404708935022717094988019787462193)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inventory(catalog):
    groups = defaultdict(list)
    identities = set()
    for entry in catalog["entries"]:
        if entry["identity_id"] in identities:
            raise ValueError("duplicate catalog identity")
        identities.add(entry["identity_id"])
        groups[entry["class_id"]].append(entry)
    singles = {c: es[0] for c, es in groups.items() if len(es) == 1}
    for class_id, entry in singles.items():
        styles = [v["style"] for v in entry["variants"]]
        if len(styles) != 4 or set(styles) != STYLES:
            raise ValueError(f"single-identity class {class_id} must have exactly four styles")
    signature = [
        [c, e["identity_id"], sorted(
            [v["style"], v.get("git_blob_sha1", "")] for v in e["variants"]
        )]
        for c, e in sorted(singles.items())
    ]
    return singles, sorted(c for c, es in groups.items() if len(es) > 1), _digest(signature)


def _allocate(group_sizes, total):
    """Largest-remainder quotas with deterministic tie breaking."""
    size = sum(group_sizes.values())
    if not 0 <= total <= size:
        raise ValueError("split count is outside the available class pool")
    if size == 0:
        return {group: 0 for group in group_sizes}
    quotas = {g: n * total // size for g, n in group_sizes.items()}
    ranked = sorted(group_sizes, key=lambda g: (-(group_sizes[g] * total % size), g))
    for group in ranked[:total - sum(quotas.values())]:
        quotas[group] += 1
    return quotas


def make_plan(catalog, curation, *, train_count=300, seed=20260912, val_count=0):
    if curation.get("source_commit", catalog["source"]["commit"]) != catalog["source"]["commit"]:
        raise ValueError("curation source commit mismatch")
    singles, reserved, fingerprint = _inventory(catalog)
    if not 0 < train_count < len(singles):
        raise ValueError("train_count must leave nonempty train and evaluation pools")
    if not 0 <= val_count < len(singles) - train_count:
        raise ValueError("val_count must leave a nonempty test pool, or be zero for combined eval")
    by_name = {entry["source_name"]: entry for entry in curation["entries"]}
    class_groups = {}
    for class_id, entry in singles.items():
        if entry["source_name"] not in by_name:
            raise ValueError(f"missing curation group for {entry['source_name']}")
        class_groups[class_id] = by_name[entry["source_name"]]["group"]
    strata = defaultdict(list)
    for class_id, group in sorted(class_groups.items()):
        strata[group].append(class_id)
    quotas = _allocate({g: len(cs) for g, cs in strata.items()}, train_count)
    train, heldout = [], {}
    for group, classes in sorted(strata.items()):
        ranked = sorted(classes, key=lambda c: (_digest([_RANK_NAMESPACE, seed, c]), c))
        train.extend(ranked[:quotas[group]])
        heldout[group] = ranked[quotas[group]:]
    splits = {"train": sorted(train)}
    if val_count:
        val_quotas = _allocate({g: len(cs) for g, cs in heldout.items()}, val_count)
        splits["val"] = sorted(c for g, cs in heldout.items() for c in cs[:val_quotas[g]])
        splits["test"] = sorted(c for g, cs in heldout.items() for c in cs[val_quotas[g]:])
    else:
        splits["eval"] = sorted(c for cs in heldout.values() for c in cs)
    return {
        "schema_version": PLAN_SCHEMA,
        "source_commit": catalog["source"]["commit"],
        "candidate_fingerprint": fingerprint,
        "selection": "Exactly one source identity per class, with all four official styles",
        "seed": seed,
        "assignment_method": "Curated broad-group stratification, largest-remainder quotas, SHA-256 class ranking; explicit class lists are authoritative",
        "eval_policy": "Separate val and test classes" if val_count else "Combined held-out pool; val/test subdivision has not been assigned",
        "class_groups": dict(sorted(class_groups.items())),
        "splits": splits,
        "reserved_classes": reserved,
        "scope_note": "Class/identity disjoint; broader semantic families may span splits. All styles and geometric augmentations inherit their class split.",
    }


def apply_plan(catalog, plan):
    singles, reserved, fingerprint = _inventory(catalog)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported split plan schema")
    if plan.get("source_commit") != catalog["source"]["commit"]:
        raise ValueError("split plan source commit mismatch")
    if plan.get("candidate_fingerprint") != fingerprint:
        raise ValueError("split plan candidate fingerprint mismatch")
    splits = plan["splits"]
    if set(splits) not in ({"train", "eval"}, {"train", "val", "test"}):
        raise ValueError("split plan must define train/eval or train/val/test")
    assignments = {}
    for split, classes in splits.items():
        if not classes:
            raise ValueError(f"empty split: {split}")
        for class_id in classes:
            if class_id in assignments:
                raise ValueError(f"class appears more than once in split plan: {class_id}")
            assignments[class_id] = split
    if set(assignments) != set(singles):
        raise ValueError("split plan must cover exactly all single-identity classes")
    if sorted(plan.get("reserved_classes", [])) != reserved:
        raise ValueError("split plan reserved classes mismatch")
    if set(plan.get("class_groups", {})) != set(singles):
        raise ValueError("split plan group metadata must cover selected classes")
    result = copy.deepcopy(catalog)
    result["entries"] = []
    for class_id, entry in sorted(singles.items()):
        selected = copy.deepcopy(entry)
        selected.update(split=assignments[class_id], split_group=class_id,
                        category_group=plan["class_groups"][class_id])
        result["entries"].append(selected)
    result["scope"] = "single_identity_four_styles"
    result["split_policy"] = plan["scope_note"]
    result["split_plan_sha256"] = _digest(plan)
    result["selection"] = {
        "classes": len(singles), "identities": len(singles),
        "assets": 4 * len(singles),
        "class_splits": {s: len(cs) for s, cs in splits.items()},
        "reserved_multi_identity_classes": len(reserved),
        "eval_policy": plan["eval_policy"],
    }
    return result


def write_selected_manifest(catalog, plan, manifest_path, asset_root):
    selected = apply_plan(catalog, plan)
    manifest_path = Path(manifest_path).resolve()
    selected["asset_root"] = os.path.relpath(Path(asset_root).resolve(), manifest_path.parent)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    return selected


def split_summary(selected):
    return {
        **selected["selection"],
        "assets_per_split": dict(sorted(Counter(e["split"] for e in selected["entries"] for _ in e["variants"]).items())),
        "groups": {
            group: dict(sorted(Counter(e["split"] for e in selected["entries"] if e["category_group"] == group).items()))
            for group in sorted({e["category_group"] for e in selected["entries"]})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=(_ROOT / 'shared/iconlib/fluent_catalog.json'))
    parser.add_argument("--curation", type=Path, default=(_ROOT / 'shared/iconlib/fluent_curation.json'))
    parser.add_argument("--split-plan", type=Path, default=(_ROOT / 'shared/iconlib/fluent_splits.json'))
    parser.add_argument("--manifest", type=Path, default=(_ROOT / 'shared/iconlib/fluent_manifest.json'))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=0)
    args = parser.parse_args()
    if args.catalog.resolve() == args.manifest.resolve():
        parser.error("catalog and selected manifest must use different paths")
    catalog = json.loads(args.catalog.read_text())
    if args.split_plan.exists():
        plan = json.loads(args.split_plan.read_text())
        if (plan["seed"] != args.seed or len(plan["splits"]["train"]) != args.train_count
                or len(plan["splits"].get("val", [])) != args.val_count):
            parser.error("existing split plan is fixed; use another --split-plan path for a different allocation")
    else:
        plan = make_plan(catalog, json.loads(args.curation.read_text()), train_count=args.train_count,
                         seed=args.seed, val_count=args.val_count)
        args.split_plan.parent.mkdir(parents=True, exist_ok=True)
        args.split_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    root = args.catalog.resolve().parent / catalog["asset_root"]
    selected = write_selected_manifest(catalog, plan, args.manifest, root)
    summary = split_summary(selected)
    (root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
