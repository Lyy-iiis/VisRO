"""The contract every task adapter and every task verifier implements.

An ADAPTER (tasks/<category>/<task>/adapter.py) turns the task's generator into
release items.  It must define:

    TASK     = "<registry name>"          # one of registry.TASKS
    VERSION  = "<task>@<release version>" # the release version, shared by all tasks
    TIER_DEFINITIONS = {tier: "one line: the exact parameters that define it"}
    def build(tier: str, seed: int, *, split: str, index: int) -> Item

`build` must be a pure function of its arguments: same (tier, seed, split,
index) -> byte-identical image and identical record, in any process.  `seed`
is already derived from (task, tier, split, base seed, index) by
seeding.derive_seed, so two tiers or two splits never share a random stream.
Raise Reject (or AssertionError / ValueError / RuntimeError) for a candidate
the generator's own gates refuse; the driver counts it and moves on.

A VERIFIER (tasks/<category>/<task>/verifier.py) checks a finished record INDEPENDENTLY
of the generator: it must not call the function that produced the answer.  It
recomputes the gold answer from the record's meta and/or from the decoded
image, and proves the answer is the only valid one.  It must define:

    CORRECTNESS = "what is recomputed, from what"
    UNIQUENESS  = "why no second answer can be valid"
    def verify(record: dict, image: "PIL.Image.Image") -> list[str]

and return a list of problems (empty == pass).  The driver runs it on every
candidate, on the image decoded back from the PNG bytes that will be written,
and rejects any candidate with a problem.  The verifier sees the full in-memory
record (with `meta`); only the public fields (PUBLIC_FIELDS) are written.
`visro verify` rebuilds each record from its candidate index and re-runs it.
"""
from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from visro.registry import SPLITS, TIERS  # noqa: F401  (part of the contract: adapters check api.SPLITS)

RELEASE_VERSION = "1.0.0"

# The fields of every written record, in this order, for every task.
PUBLIC_FIELDS = ("id", "task", "category", "split", "difficulty", "question", "answer", "answer_type",
                 "options", "image", "image_sha256", "image_size")
PUBLIC_ANSWER_TYPES = ("choice", "free_form")

# Answer types used across the release.  `choice` answers are one of `choices`.
ANSWER_TYPES = (
    "int",          # "7"
    "choice",       # "B"  (with choices=["A","B","C","D"])
    "bool",         # "yes" / "no"
    "bitstring",    # "0110"
    "token-list",   # "6,4,2" / "B3,C3"
    "pair-list",    # "a-b,c-d"
    "text",         # anything else; the verifier states the format
)


class Reject(Exception):
    """A candidate the adapter's own gates refuse.  Counted, never fatal."""


class AssetError(Exception):
    """Required assets are absent or differ from their pinned hashes.  Fatal: not in REJECT_EXCEPTIONS,
    so generation stops instead of skipping the candidate."""


REJECT_EXCEPTIONS = (Reject, AssertionError, ValueError, RuntimeError)


@dataclasses.dataclass
class Item:
    question: str
    answer: str
    image: Any                      # PIL.Image.Image, the question image
    meta: dict                      # everything the verifier needs; JSON-serialisable
    core_key: str                   # content-canonical dedup key (sha256 hex)
    answer_type: str
    choices: list | None = None
    options: list | None = None     # options shown with a free-form question (e.g. suggested values)

    def public_answer_type(self) -> str:
        return "choice" if self.answer_type == "choice" else "free_form"

    def public_options(self) -> list:
        if self.answer_type == "choice":
            return [str(c) for c in self.choices]
        return [str(o) for o in (self.options or [])]

    def validate(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("empty question")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"unknown answer_type {self.answer_type!r}")
        if self.answer_type == "choice":
            if not self.choices or self.answer not in self.choices:
                raise ValueError("choice answer not among choices")
        if not isinstance(self.core_key, str) or len(self.core_key) < 16:
            raise ValueError("core_key must be a hex digest")


def _load(kind: str, task: str):
    from visro.registry import module_name
    return importlib.import_module(module_name(task, kind))


def load_adapter(task: str):
    mod = _load("adapters", task)
    for attr in ("TASK", "VERSION", "TIER_DEFINITIONS", "build"):
        if not hasattr(mod, attr):
            raise AttributeError(f"adapter {task} lacks {attr}")
    if mod.TASK != task:
        raise ValueError(f"adapter {task} declares TASK={mod.TASK!r}")
    if tuple(mod.TIER_DEFINITIONS) != TIERS:
        raise ValueError(f"adapter {task} TIER_DEFINITIONS must list {TIERS} in order")
    return mod


def load_verifier(task: str):
    mod = _load("verifiers", task)
    for attr in ("CORRECTNESS", "UNIQUENESS", "verify"):
        if not hasattr(mod, attr):
            raise AttributeError(f"verifier {task} lacks {attr}")
    return mod

