"""The 24 tasks, grouped into the four categories of the VisRO paper (Appendix E.1), and the
tiers and splits every task is generated in.

Standard library only: the light reader and scorer (visro.dataset, visro.scoring) import it.
"""
from __future__ import annotations

CATEGORIES = {
    "discrimination_grouping": ("count_clusters", "count_points_on_lines", "find_the_different", "find_the_same",
                                "find_the_shadow", "overlap_match", "overlapping_glyphs"),
    "paths_connectivity": ("flow_free", "maze", "metro_map", "path_tracing"),
    "planar_transformations": ("jigsaw_completion", "logic_matrix", "mirroring_patterns", "reconstruction",
                               "rotation_patterns", "tangram_reconstruction"),
    "spatial_transformations": ("block_delta", "count_3d_blocks", "cube_unfold", "paper_folding", "three_d_views",
                                "view_reconstruction", "viewpoint_rotation"),
}
TASKS = tuple(sorted(t for tasks in CATEGORIES.values() for t in tasks))
CATEGORY_OF = {t: c for c, tasks in CATEGORIES.items() for t in tasks}
CORE6 = ("count_clusters", "flow_free", "maze", "metro_map", "overlapping_glyphs", "path_tracing")
ALIASES = {"flow_free_count": "flow_free"}

TIERS = ("easy", "medium", "hard", "xhard", "xxhard")
# `release` generates the splits in this order, each disjoint from the ones before it.
SPLITS = ("train", "val", "eval", "ood")
# The published dataset has no val split.
PUBLISHED_SPLITS = ("train", "eval", "ood")


def resolve(name: str) -> str:
    if name in CATEGORY_OF:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(f"unknown task {name!r}; known: {', '.join(TASKS)}")


def select(names: list[str] | None) -> list[str]:
    """Expand task names, category names and 'all' into registry order."""
    if not names or names == ["all"]:
        return list(TASKS)
    out: list[str] = []
    for n in names:
        if n in CATEGORIES:
            out += [t for t in TASKS if CATEGORY_OF[t] == n]
        else:
            out.append(resolve(n))
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def category(task: str) -> str:
    return CATEGORY_OF[resolve(task)]


def task_dir(task):
    from .paths import ROOT
    task = resolve(task)
    return ROOT / 'tasks' / CATEGORY_OF[task] / task


def module_name(task, kind):
    task = resolve(task)
    leaf = {'adapters': 'adapter', 'verifiers': 'verifier'}[kind]
    return f"tasks.{CATEGORY_OF[task]}.{task}.{leaf}"
