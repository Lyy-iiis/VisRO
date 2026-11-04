"""Fold/punch geometry of the easy/medium/hard tiers (one, two and three folds): CONFIG and all_plans()."""

from . import folding

CONFIG = {
    'easy': dict(n=8, folds=1, stages=[1], punches=[4]),
    'medium': dict(n=8, folds=2, stages=[1, 2], punches=[2, 2]),
    'hard': dict(n=8, folds=3, stages=[1, 3], punches=[2, 2]),
}


class Branch(Exception):
    def __init__(self, choices):
        self.choices = choices


class Replay:
    def __init__(self, prefix):
        self.items = iter(prefix)

    def choice(self, options):
        try:
            v = next(self.items)
        except StopIteration:
            raise Branch(options)
        assert v in options
        return v


def all_plans(n, count, diagonal):
    pending = [[]]
    out = []
    while pending:
        prefix = pending.pop()
        try:
            out.append(folding.fold_plan(n, count, Replay(prefix), diagonal))
        except Branch as b:
            pending.extend(prefix + [v] for v in b.choices)
    return sorted(out, key=str)
