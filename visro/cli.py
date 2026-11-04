"""python -m visro {list,docs,generate,verify,release,preview,score}

Run from the distribution root.  Each task runs in its own
process because many generators share module names (generate.py, render.py).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from visro import registry
from visro.api import RELEASE_VERSION
from visro.paths import ROOT
from visro.registry import SPLITS, TIERS


def _path(value):
    return Path(value).expanduser().resolve()


def _tiers(arg: str | None) -> list[str]:
    if not arg or arg == "all":
        return list(TIERS)
    out = [t.strip() for t in arg.split(",") if t.strip()]
    bad = [t for t in out if t not in TIERS]
    if bad:
        raise SystemExit(f"unknown tier(s) {bad}; tiers are {TIERS}")
    return out


def _describe(task: str) -> dict:
    from visro.api import load_adapter, load_verifier
    ad, ve = load_adapter(task), load_verifier(task)
    return {"task": task, "category": registry.category(task), "version": RELEASE_VERSION,
            "tiers": {t: ad.TIER_DEFINITIONS[t] for t in TIERS},
            "correctness": ve.CORRECTNESS, "uniqueness": ve.UNIQUENESS}


def cmd_list(a):
    from visro.api import load_adapter, load_verifier
    if a.json:
        if _per_task(a, "list", ["--json"]):
            return
        print(json.dumps(_describe(registry.select(a.tasks)[0]), ensure_ascii=False))
        return
    tasks = registry.select(a.tasks)
    task_width, category_width = max(map(len, tasks)), max(map(len, registry.CATEGORIES))
    for task in tasks:
        line = f"{task:{task_width}s} {registry.category(task):{category_width}s}"
        if a.detail:
            try:
                ad = load_adapter(task)
                ve = load_verifier(task)
                print(line + f" {RELEASE_VERSION}")
                for tier in TIERS:
                    print(f"    {tier:7s} {ad.TIER_DEFINITIONS[tier]}")
                print(f"    correctness: {ve.CORRECTNESS}\n    uniqueness:  {ve.UNIQUENESS}")
            except Exception as exc:  # noqa: BLE001
                print(line + f" (not loadable: {exc!r})")
        else:
            print(line.rstrip())


def _per_task(a, sub: str, extra: list[str]):
    """Fan out one subprocess per task, sequentially, and fail if any fails."""
    tasks = registry.select(a.tasks)
    if len(tasks) == 1:
        return None
    failed = []
    for task in tasks:
        r = subprocess.run([sys.executable, "-m", "visro", sub, "--tasks", task, *extra], cwd=ROOT)
        if r.returncode:
            failed.append(task)
    if failed:
        raise SystemExit(f"{sub} failed for: {', '.join(failed)}")
    return True


def cmd_generate(a):
    common = ["--out", str(a.out), "--n", str(a.n), "--seed", str(a.seed), "--split", a.split,
              "--tiers", ",".join(_tiers(a.tiers)), "--workers", str(a.workers)]
    if a.overwrite:
        common.append("--overwrite")
    if a.no_verify:
        common.append("--no-verify")
    for d in a.exclude_keys_from or []:
        common += ["--exclude-keys-from", str(d)]
    if _per_task(a, "generate", common):
        return
    from visro.driver import generate_cell, load_seen
    task = registry.select(a.tasks)[0]
    seen = load_seen(a.exclude_keys_from or [], task)
    stats = []
    for tier in _tiers(a.tiers):  # one seen-set across tiers: no item repeats across a task's tiers
        stats.append(generate_cell(task, tier, Path(a.out), n=a.n, seed=a.seed, split=a.split,
                                   workers=a.workers, seen=seen, verify=not a.no_verify,
                                   overwrite=a.overwrite))
    print(json.dumps({"task": task, "cells": [{k: s[k] for k in ("tier", "accepted", "candidates_consumed")}
                                              for s in stats]}))


def cmd_verify(a):
    absent = [str(r) for r in a.root if not Path(r).is_dir()]
    if absent:
        raise SystemExit(f"--root does not exist: {', '.join(absent)}")
    extra = ["--rebuild", str(a.rebuild), "--tiers", ",".join(_tiers(a.tiers))]
    for r in a.root:
        extra += ["--root", str(r)]
    if a.sample is not None:
        extra += ["--sample", str(a.sample)]
    if a.allow_missing:
        extra.append("--allow-missing")
    if _per_task(a, "verify", extra):
        return
    from visro.checks import check_cells
    task = registry.select(a.tasks)[0]
    # Several roots (e.g. train/ val/ eval/ ood/) are checked together, so a
    # duplicate across splits is reported like a duplicate within a cell.
    # Every (root, tier) cell must exist unless --allow-missing.
    cells = [Path(r) / f"{task}_{t}" for r in a.root for t in _tiers(a.tiers)]
    missing = [c for c in cells if not (c / "records.jsonl").exists()]
    if missing and not a.allow_missing:
        raise SystemExit(f"{task}: missing cells (no records.jsonl; --allow-missing skips them): "
                         f"{', '.join(str(c) for c in missing)}")
    for c in missing:
        print(f"SKIPPED {c} (no records.jsonl)")
    cells = [c for c in cells if c not in missing]
    if not cells:
        raise SystemExit(f"{task}: no cell to verify under {', '.join(str(r) for r in a.root)}")
    rep = check_cells(cells, verify_sample=a.sample, rebuild_sample=a.rebuild)
    for name, c in rep["cells"].items():
        print(f"{name}: records={c['records']} verified={c['verified']} rebuilt={c['rebuilt']} problems={c['problems']}")
    for p in rep["problems"][:50]:
        print("  PROBLEM", p)
    if rep["problems"]:
        raise SystemExit(f"{task}: {len(rep['problems'])} problems")


def cmd_release(a):
    """Generate every requested split under <out>/<split>/, in registry.SPLITS order,
    each split excluding the items and images of the splits before it."""
    done = []
    for split in SPLITS:
        n = getattr(a, f"n_{split}")
        if not n:
            continue
        g = argparse.Namespace(tasks=a.tasks, tiers=a.tiers, out=Path(a.out) / split, n=n, seed=a.seed,
                               split=split, workers=a.workers, overwrite=a.overwrite, no_verify=False,
                               exclude_keys_from=[Path(a.out) / d for d in done])
        print(f"== {split}: {n} per cell, excluding {done or 'nothing'}", flush=True)
        cmd_generate(g)
        done.append(split)


# Characters that GitHub-flavoured Markdown would drop or reinterpret in plain
# text: '*' (emphasis: "(4*index+2) % 7" would lose its star), '_' where it can
# delimit emphasis (not between two letters or digits), code spans, raw HTML
# and autolinks ('<'), link brackets, entities, strikethrough, table pipes,
# math, and the backslash itself.  GFM renders a backslash-escaped ASCII
# punctuation character as that character.
_MD_ACTIVE = re.compile(r"[\\`*<\[\]&~|$]|(?<![^\W_])_|_(?![^\W_])")


def md_text(s: str) -> str:
    """Plain text -> GFM source that renders as exactly that text, in a table cell or a paragraph."""
    if "\n" in s or "\r" in s:
        raise ValueError(f"line break in a one-line description: {s[:80]!r}")
    return _MD_ACTIVE.sub(lambda m: "\\" + m.group(0), s)


def cmd_docs(a):
    """Write the task/tier reference (TASKS.md) from the adapters and verifiers themselves."""
    rows, sections = [], []
    for task in registry.select(a.tasks):
        r = subprocess.run([sys.executable, "-m", "visro", "list", "--json", "--tasks", task], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{task}: cannot load adapter/verifier:\n{r.stderr[-2000:]}")
        d = json.loads(r.stdout.strip().splitlines()[-1])
        readme = f"../tasks/{d['category']}/{task}/README.md"
        rows.append(f"| [{task}](#{task}) | {d['category']} | [{readme}]({readme}) |")
        cells = {t: md_text(d["tiers"][t]) for t in TIERS}
        tiers = "\n".join(f"| {t} | {cells[t]} |" for t in TIERS)
        sections.append(f"## {task}\n\ncategory {d['category']} &middot; details in [{readme}]({readme})\n\n"
                        f"| tier | definition |\n|---|---|\n{tiers}\n\n"
                        f"**Correctness check.** {md_text(d['correctness'])}\n\n"
                        f"**Uniqueness check.** {md_text(d['uniqueness'])}\n")
    text = ("# Tasks and tiers\n\nGenerated by `python -m visro docs` from each task's `TIER_DEFINITIONS` and "
            "its verifier's `CORRECTNESS` / `UNIQUENESS` texts; do not edit by hand.\n\n"
            f"Release {RELEASE_VERSION}: {len(rows)} tasks x {len(TIERS)} tiers ({', '.join(TIERS)}).\n\n"
            "| task | category | notes |\n|---|---|---|\n" + "\n".join(rows) + "\n\n" +
            "\n".join(sections))
    Path(a.out).write_text(text)
    print(a.out)


def cmd_preview(a):
    from visro.preview import build_preview
    index = build_preview(Path(a.root), Path(a.out), per_tier=a.per_tier,
                          tasks=registry.select(a.tasks) if a.tasks else None, verify=not a.no_verify)
    print(index)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["score"]:
        from visro.scoring import main as score_main
        return score_main(argv[1:], prog="python -m visro score")
    p = argparse.ArgumentParser(prog="visro", epilog="score: python -m visro score --help")
    sp = p.add_subparsers(dest="cmd", required=True)

    q = sp.add_parser("list")
    q.add_argument("--tasks", nargs="*")
    q.add_argument("--detail", action="store_true")
    q.add_argument("--json", action="store_true", help="one JSON line per task (tiers, verifier texts)")
    q.set_defaults(fn=cmd_list)

    q = sp.add_parser("docs", help="write TASKS.md from the adapters and verifiers")
    q.add_argument("--tasks", nargs="*", default=["all"])
    q.add_argument("--out", type=_path, default=(ROOT / 'docs/TASKS.md'))
    q.set_defaults(fn=cmd_docs)

    q = sp.add_parser("generate")
    q.add_argument("--tasks", nargs="*", default=["all"])
    q.add_argument("--tiers", default="all")
    q.add_argument("--out", required=True, type=_path)
    q.add_argument("--n", type=int, required=True, help="records per cell")
    q.add_argument("--seed", type=int, default=20261011)
    q.add_argument("--split", choices=SPLITS, default="train")
    q.add_argument("--workers", type=int, default=1)
    q.add_argument("--exclude-keys-from", action="append", type=_path,
                   help="cells whose items and images must not be reused (e.g. train, when making eval)")
    q.add_argument("--no-verify", action="store_true", help="skip the independent verifier gate (not for release)")
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(fn=cmd_generate)

    q = sp.add_parser("verify")
    q.add_argument("--tasks", nargs="*", default=["all"])
    q.add_argument("--tiers", default="all")
    q.add_argument("--root", required=True, type=_path, action="append",
                   help="cell root; repeat to check several splits together (cross-split duplicates)")
    q.add_argument("--sample", type=int, default=None,
                   help="rebuild and verify a sample per cell (default: every record)")
    q.add_argument("--rebuild", type=int, default=3,
                   help="at least this many records per cell rebuilt and compared byte for byte")
    q.add_argument("--allow-missing", action="store_true",
                   help="skip (task, tier) cells absent under a root instead of failing; a task with no cell still fails")
    q.set_defaults(fn=cmd_verify)

    q = sp.add_parser("release", help="train/val/eval/ood under one root, disjoint by item and image")
    q.add_argument("--tasks", nargs="*", default=["all"])
    q.add_argument("--tiers", default="all")
    q.add_argument("--out", required=True, type=_path)
    q.add_argument("--seed", type=int, default=20261011)
    q.add_argument("--workers", type=int, default=1)
    for split in SPLITS:
        q.add_argument(f"--n-{split}", type=int, default=0, help=f"records per cell in {split} (0 = skip)")
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(fn=cmd_release)

    q = sp.add_parser("preview")
    q.add_argument("--tasks", nargs="*")
    q.add_argument("--root", required=True, type=_path)
    q.add_argument("--out", required=True, type=_path)
    q.add_argument("--per-tier", type=int, default=3)
    q.add_argument("--no-verify", action="store_true")
    q.set_defaults(fn=cmd_preview)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
