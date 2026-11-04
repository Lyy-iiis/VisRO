"""Score predictions on VisRO: one rule per task, using only the public record fields.

    from visro.scoring import score_record, summarize

    result = score_record(record, "B")      # {"correct": bool, "parsed": ..., "gold": ..., "rule": str}

    python -m visro score --root VisRO --predictions preds.jsonl [--tasks core6] [--split eval]

A record is a published row (``task``, ``answer_type``, ``options``, ``question``, ``answer``; nothing
else is read).  A prediction is the model's answer text.  Every rule parses the prediction into the
task's answer and compares it with the parsed reference answer.

Core6 (count_clusters, flow_free, maze, metro_map, overlapping_glyphs, path_tracing) follows the
scoring rules of the paper's experiments.  Where a rule needs more than the published fields, the
extra input is derived from the question text (docs/DATASET.md):

* maze: the answer kind -- ``choice`` (answer_type), ``boolean`` (the question asks for True or
  False) or ``number`` (otherwise).  choice: the first character of the prediction when it is one
  of A-F, else the first of A..F (in that order) that stands alone in it; boolean: the first yes/no/true/false/1/0 token;
  number: the first integer.
* path_tracing: the question type -- ``one_marker``, ``ordered``, ``all_pairs`` or ``same_line``
  (one fixed question template each) -- and, for ``ordered``, the marker labels of the scene,
  needed to split an unseparated answer such as ``541210``: letters A-Z and the numbers 1 to the
  larger of 9 and the position of the last start letter the question names.  one_marker: the
  label; ordered: the label sequence, which must split one way only; all_pairs: the set of
  letter-number pairs; same_line: yes/no.
* metro_map: a question that asks for a station name is scored on the name (case-insensitive,
  surrounding quotes and a trailing period ignored); every other question on its one integer.
* count_clusters: exactly one integer.  flow_free: the integer list, in order.
  overlapping_glyphs: the letters and digits, in order.

The other 18 tasks: the whole cleaned prediction must be one complete answer of the format the
question states (cleaning: surrounding whitespace, a leading ``answer:``, quotes, backticks,
``**bold**``, ``\\boxed{}``, a trailing period; a multi-line prediction is tried whole, then by
its first non-empty line):

* choice tasks (overlap_match, reconstruction, three_d_views, view_reconstruction,
  viewpoint_rotation; logic_matrix choice items): one letter A-H, which must be an option;
* find_the_same: the set of unordered index pairs; cube_unfold: the set of cube numbers;
  find_the_different, block_delta position questions: the ordered pair; block_delta /
  count_3d_blocks counts: one integer;
* count_points_on_lines (``a,b,..;total``), find_the_shadow (``A-n,..;unmatched:..``) and
  jigsaw_completion (``X1,Y0,..``): the exact string with whitespace removed (case included);
* logic_matrix grids (``0101/..``), paper_folding bit strings, mirroring_patterns colour codes
  (R B G Y P W; spaces or commas between codes are allowed), rotation_patterns figure numbers or
  ``none`` in order, tangram_reconstruction ``[[x,y,k],..]`` in piece order.

Standard library only (plus ``visro.registry``, and ``visro.dataset`` for the command line).
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping

from visro.registry import CORE6, PUBLISHED_SPLITS


# ============================================================================ core6 (paper scorer)
_BOOL_TRUE = {"true", "yes", "y", "1"}
_BOOL_FALSE = {"false", "no", "n", "0"}


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip("`'\"")
    text = text.replace("\\boxed{", "").replace("}", "")
    return text.strip()


def _normalize_bool(text: Any) -> bool | None:
    for token in re.findall(r"[a-z0-9]+", _normalize_text(text).lower()):
        if token in _BOOL_TRUE:
            return True
        if token in _BOOL_FALSE:
            return False
    return None


def _marker_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _integers(text: Any) -> list[int]:
    return [int(token) for token in re.findall(r"-?\d+", _normalize_text(text))]


def _station(text: Any) -> str:
    return _normalize_text(text).rstrip(".").strip().strip("`'\"").casefold()


def _pairs(text: Any) -> set[tuple[str, str]]:
    raw = _normalize_text(text).upper()
    pairs = set()
    for a, b in re.findall(r"([A-Z]+)\s*[-:=>]+\s*([0-9]+)", raw):
        pairs.add((_marker_token(a), _marker_token(b)))
    if pairs:
        return pairs
    tokens = [_marker_token(t) for t in re.findall(r"[A-Z]+|[0-9]+", raw)]
    for a, b in zip(tokens, tokens[1:]):
        if re.fullmatch(r"[A-Z]+", a) and re.fullmatch(r"[0-9]+", b):
            pairs.add((a, b))
    return pairs


def _label_sequence(text: Any, vocabulary: set[str]) -> list[str]:
    """Labels separated by anything; an unseparated run is split over ``vocabulary`` and must split
    one way only (else no answer)."""
    chunks = re.findall(r"[A-Z0-9]+", _normalize_text(text).upper())
    if len(chunks) != 1:
        return chunks if all(token in vocabulary for token in chunks) else []
    compact = chunks[0]
    ways = [0] * (len(compact) + 1)
    next_token = [None] * len(compact)
    ways[-1] = 1
    for offset in range(len(compact) - 1, -1, -1):
        for token in vocabulary:
            end = offset + len(token)
            if compact.startswith(token, offset) and ways[end]:
                ways[offset] = min(2, ways[offset] + ways[end])
                next_token[offset] = token
                if ways[offset] == 2:
                    break
    if ways[0] != 1:
        return []
    parsed, offset = [], 0
    while offset < len(compact):
        parsed.append(next_token[offset])
        offset += len(next_token[offset])
    return parsed


def maze_kind(record: Mapping[str, Any]) -> str:
    """``choice`` / ``boolean`` / ``number``."""
    if record.get("answer_type") == "choice":
        return "choice"
    return "boolean" if re.search(r"\bTrue or False\b", record["question"]) else "number"


_PATH_TYPES = (
    ("ordered", re.compile(r"What is ([A-Z]+) connected to, in that order\?")),
    ("one_marker", re.compile(r"What is ([A-Z0-9]+) connected to\?")),
    ("all_pairs", re.compile(r"List all connected pairs\.")),
    ("same_line", re.compile(r"Are the two colored dots connected by the same line\?")),
)


def path_question_type(question: str) -> str:
    """``one_marker`` / ``ordered`` / ``all_pairs`` / ``same_line``."""
    for name, pattern in _PATH_TYPES:
        if pattern.match(question):
            return name
    raise ValueError(f"not a path_tracing question: {question!r}")


def path_vocabulary(question: str) -> set[str]:
    """Labels of an ``ordered`` question's scene: A-Z and 1..max(9, position of the last named start)."""
    starts = _PATH_TYPES[0][1].match(question).group(1)
    last = max(string.ascii_uppercase.index(c) + 1 for c in starts)
    return set(string.ascii_uppercase) | {str(i) for i in range(1, max(9, last) + 1)}


def metro_asks_station(question: str) -> bool:
    return re.search(r"\bstation name\b", question) is not None


def _check_choice(prediction: Any, gold: str) -> bool:
    pred = str(prediction).strip().upper()
    if pred and pred[0] in "ABCDEF":
        return pred[0] == gold
    for letter in "ABCDEF":
        if re.search(rf"\b{letter}\b", pred):
            return letter == gold
    return False


def _score_core6(record: Mapping[str, Any], prediction: Any) -> dict[str, Any]:
    task, answer, question = record["task"], str(record["answer"]).strip(), record["question"]
    if task == "maze":
        kind = maze_kind(record)
        if kind == "choice":
            gold = answer.upper()
            return {"correct": _check_choice(prediction, gold), "parsed": None, "gold": gold, "rule": "maze:choice"}
        if kind == "boolean":
            parsed, gold = _normalize_bool(prediction), _normalize_bool(answer)
            return {"correct": parsed is not None and gold is not None and parsed == gold,
                    "parsed": parsed, "gold": gold, "rule": "maze:boolean"}
        tokens = re.findall(r"-?\d+", _normalize_text(prediction))
        return {"correct": bool(tokens) and tokens[0] == answer, "parsed": tokens[0] if tokens else None,
                "gold": answer, "rule": "maze:number"}
    if task == "path_tracing":
        kind = path_question_type(question)
        if kind == "one_marker":
            parsed, gold = _marker_token(_normalize_text(prediction)), _marker_token(_normalize_text(answer))
            correct = parsed == gold
        elif kind == "ordered":
            vocabulary = path_vocabulary(question)
            parsed, gold = _label_sequence(prediction, vocabulary), _label_sequence(answer, vocabulary)
            correct = bool(gold) and parsed == gold
        elif kind == "all_pairs":
            parsed, gold = sorted(_pairs(prediction)), sorted(_pairs(answer))
            correct = parsed == gold
        else:
            parsed, gold = _normalize_bool(prediction), _normalize_bool(answer)
            correct = parsed is not None and gold is not None and parsed == gold
        return {"correct": correct, "parsed": parsed, "gold": gold, "rule": f"path_tracing:{kind}"}
    if task == "metro_map" and metro_asks_station(question):
        parsed, gold = _station(prediction), _station(answer)
        return {"correct": bool(gold) and parsed == gold, "parsed": parsed, "gold": gold, "rule": "metro_map:station"}
    if task == "overlapping_glyphs":
        parsed = re.findall(r"[A-Z0-9]", _normalize_text(prediction).upper())
        gold = re.findall(r"[A-Z0-9]", _normalize_text(answer).upper())
        return {"correct": bool(gold) and parsed == gold, "parsed": parsed, "gold": gold, "rule": "glyphs"}
    parsed, gold = _integers(prediction), _integers(answer)
    if task == "flow_free":
        return {"correct": bool(gold) and parsed == gold, "parsed": parsed, "gold": gold, "rule": "integer-list"}
    return {"correct": len(parsed) == 1 and parsed == gold, "parsed": parsed, "gold": gold,
            "rule": "integer"}           # count_clusters, metro_map counts


# ============================================================================ the other 18 tasks
class _AnswerFormatError(ValueError):
    """The text is not one complete answer of the task's format."""


def _strip_answer_prefix(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("`'\"")


def _strip_markup(text: Any) -> str:
    text = str(text).strip().strip("`").strip()
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text).strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[7:-1]
    return text.strip().rstrip(".").strip()


def _clean(value: Any) -> str:
    return _strip_markup(_strip_answer_prefix(value))


def _pair_set(value: Any) -> list[list[int]]:
    text = _strip_answer_prefix(value).rstrip(".").replace("–", "-").replace("—", "-")
    if text.startswith("["):
        try:
            pairs = json.loads(text)
        except json.JSONDecodeError as error:
            raise _AnswerFormatError("invalid JSON pair set") from error
    elif re.fullmatch(r"\d+\s*-\s*\d+(?:\s*[,;\n]\s*\d+\s*-\s*\d+)*", text):
        pairs = [list(map(int, match)) for match in re.findall(r"(\d+)\s*-\s*(\d+)", text)]
    else:
        raise _AnswerFormatError("expected a complete pair list such as 1-7,2-9")
    if not isinstance(pairs, (list, tuple)):
        raise _AnswerFormatError("pair set must be a list")
    result = set()
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(x) is not int or x < 1 for x in pair) or pair[0] == pair[1]):
            raise _AnswerFormatError("each pair needs two different positive integer indices")
        result.add(tuple(sorted(pair)))
    return [list(pair) for pair in sorted(result)]


_CANONICAL_TASKS = {"find_the_same", "three_d_views", "view_reconstruction", "viewpoint_rotation", "cube_unfold",
                    "find_the_different", "block_delta", "count_3d_blocks"}


def _canonical_answer(task: str, value: Any):
    """The answer of eight tasks as a canonical value (pair set, letter, number set, ordered pair or
    integer); it strips markup from the (already cleaned) text once more."""
    value = _strip_markup(value)
    if task == "find_the_same":
        return _pair_set(value)
    if task in ("three_d_views", "view_reconstruction", "viewpoint_rotation"):
        if not re.fullmatch("[A-Ha-h]", value):
            raise _AnswerFormatError("not a single choice")
        return value.upper()
    if task == "cube_unfold":
        if not re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", value):
            raise _AnswerFormatError("not a complete number list")
        nums = [int(n) for n in re.findall(r"\d+", value)]
        if min(nums) < 1:
            raise _AnswerFormatError("nonpositive candidate index")
        return sorted(set(nums))
    if task == "find_the_different" or (task == "block_delta" and "," in value):
        if (value.startswith("(") and value.endswith(")")) or (value.startswith("[") and value.endswith("]")):
            value = value[1:-1]
        if not re.fullmatch(r"\d+\s*,\s*\d+", value):
            raise _AnswerFormatError("not an ordered pair")
        return tuple(map(int, value.split(",")))
    if not re.fullmatch(r"[+-]?\d+", value):
        raise _AnswerFormatError("not a single integer")
    return int(value)


def _option_letter(value: str, options: Iterable[Any]) -> str:
    if not re.fullmatch("[A-Ha-h]", value.strip()):
        raise _AnswerFormatError(f"not a single option letter: {value!r}")
    letter = value.strip().upper()
    if letter not in {str(o).strip().upper() for o in options}:
        raise _AnswerFormatError(f"{letter} is not an option")
    return letter


def _nows(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _exact(pattern: str, what: str):
    def rule(value: str):
        s = _nows(value)
        if not re.fullmatch(pattern, s):
            raise _AnswerFormatError(f"not {what}: {value!r}")
        return s
    return rule


def _jigsaw(value: str) -> tuple[str, ...]:
    s = _nows(value)
    if not re.fullmatch(r"[A-Z][0-3](?:,[A-Z][0-3])*", s):
        raise _AnswerFormatError(f"not 'X1,Y0,..': {value!r}")
    return tuple(s.split(","))


def _grid(value: str) -> tuple[str, ...]:
    s = _nows(value)
    if not re.fullmatch(r"[01]+(?:/[01]+)*", s):
        raise _AnswerFormatError(f"not '0101/..' rows: {value!r}")
    return tuple(s.split("/"))


def _colours(value: str) -> str:
    s = value.strip().upper()
    if not re.fullmatch(r"[RBGYPW](?:[\s,]*[RBGYPW])*", s):
        raise _AnswerFormatError(f"not a string of colour codes R/B/G/Y/P/W: {value!r}")
    return re.sub(r"[\s,]", "", s)


def _figures(value: str) -> tuple[str, ...]:
    s = _nows(value).lower()
    if not re.fullmatch(r"(?:\d+|none)(?:,(?:\d+|none))*", s):
        raise _AnswerFormatError(f"not 'a,b,none,..': {value!r}")
    return tuple(str(int(t)) if t.isdigit() else t for t in s.split(","))


def _placements(value: str) -> tuple[tuple[int, int, int], ...]:
    s = _nows(value)
    if not re.fullmatch(r"\[\[\d+,\d+,\d+\](?:,\[\d+,\d+,\d+\])*\]", s):
        raise _AnswerFormatError(f"not '[[x,y,k],...]': {value!r}")
    return tuple(tuple(int(v) for v in t) for t in json.loads(s))


_CHOICE_TASKS = {"overlap_match", "reconstruction"}
_RULES = {
    "count_points_on_lines": ("counts;total", lambda v, r: _exact(r"\d+(?:,\d+)*;\d+", "'a,b,..;total'")(v)),
    "find_the_shadow": ("pairs;unmatched", lambda v, r: _exact(
        r"[A-Z]-\d+(?:,[A-Z]-\d+)*;unmatched:(?:\d+(?:,\d+)*)?", "'A-n,..;unmatched:..'")(v)),
    "jigsaw_completion": ("pieces", lambda v, r: _jigsaw(v)),
    "mirroring_patterns": ("colour-codes", lambda v, r: _colours(v)),
    "paper_folding": ("bitstring", lambda v, r: _exact(r"[01]+", "a bit string")(v)),
    "rotation_patterns": ("figure-list", lambda v, r: _figures(v)),
    "tangram_reconstruction": ("placements", lambda v, r: _placements(v)),
}


def _rule(record: Mapping[str, Any]):
    task = record["task"]
    if task in _CANONICAL_TASKS:
        name = {"find_the_same": "pairs", "cube_unfold": "index-set", "find_the_different": "ordered-pair",
                "block_delta": "pair-or-integer", "count_3d_blocks": "integer"}.get(task, "choice")
        return name, lambda v, r: _canonical_answer(task, v)
    if task in _CHOICE_TASKS or (task == "logic_matrix" and record.get("answer_type") == "choice"):
        return "choice", lambda v, r: _option_letter(v, r.get("options") or [])
    if task == "logic_matrix":
        return "binary-grid", lambda v, r: _grid(v)
    if task in _RULES:
        return _RULES[task]
    raise ValueError(f"unknown task {task!r}")


def _parse(value: Any, record: Mapping[str, Any], parse) -> Any:
    raw = "" if value is None else str(value)
    tries = [raw]
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        tries.append(lines[0])
    error = None
    for text in tries:
        try:
            return parse(_clean(text), record)
        except (ValueError, TypeError) as exc:
            error = exc
    raise ValueError(str(error))


def _plain(x: Any) -> Any:
    return [_plain(v) for v in x] if isinstance(x, (list, tuple)) else x


def _score_other(record: Mapping[str, Any], prediction: Any) -> dict[str, Any]:
    name, parse = _rule(record)
    gold = _parse(record["answer"], record, parse)          # a reference answer that does not parse is a data bug
    try:
        parsed = _parse(prediction, record, parse)
    except ValueError:
        return {"correct": False, "parsed": None, "gold": _plain(gold), "rule": name}
    return {"correct": parsed == gold, "parsed": _plain(parsed), "gold": _plain(gold), "rule": name}


# ============================================================================ public API
def score_record(record: Mapping[str, Any], prediction: Any) -> dict[str, Any]:
    """``{"correct", "parsed", "gold", "rule"}`` for one published record and one prediction."""
    if record["task"] in CORE6:
        return _score_core6(record, prediction)
    return _score_other(record, prediction)


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per task and per task x difficulty accuracy, the macro mean over tasks and the micro mean.
    ``rows`` carry ``task``, ``difficulty`` and ``correct``."""
    by_task = defaultdict(lambda: [0, 0])
    by_level = defaultdict(lambda: [0, 0])
    for row in rows:
        for key, table in ((row["task"], by_task), ((row["task"], row["difficulty"]), by_level)):
            table[key][0] += int(bool(row["correct"]))
            table[key][1] += 1
    tasks = {t: {"correct": c, "n": n, "accuracy": c / n} for t, (c, n) in sorted(by_task.items())}
    for (t, level), (c, n) in sorted(by_level.items()):
        tasks[t].setdefault("by_difficulty", {})[level] = {"correct": c, "n": n, "accuracy": c / n}
    total = sum(v["correct"] for v in tasks.values())
    count = sum(v["n"] for v in tasks.values())
    return {"tasks": tasks,
            "macro_accuracy": sum(v["accuracy"] for v in tasks.values()) / len(tasks) if tasks else 0.0,
            "micro_accuracy": total / count if count else 0.0, "correct": total, "n": count}


def load_predictions(path: str) -> dict[str, str]:
    """``{id: prediction}`` from a JSONL file (or a JSON list) of ``{"id", "prediction"}`` objects."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    out = {}
    for row in rows:
        if row["id"] in out:
            raise ValueError(f"{path}: duplicate prediction for {row['id']}")
        out[row["id"]] = row["prediction"]
    return out


def score_records(records: Iterable[Mapping[str, Any]], predictions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-item rows; a record without a prediction is scored wrong and flagged ``missing``."""
    rows = []
    for rec in records:
        missing = rec["id"] not in predictions
        result = score_record(rec, "" if missing else predictions[rec["id"]])
        rows.append({"id": rec["id"], "task": rec["task"], "split": rec["split"], "difficulty": rec["difficulty"],
                     "prediction": None if missing else predictions[rec["id"]], "missing": missing, **result})
    return rows


def main(argv=None, prog="visro-score"):
    ap = argparse.ArgumentParser(prog=prog, description="Score predictions on VisRO.")
    ap.add_argument("--root", required=True, help="the downloaded dataset (holds tasks/)")
    ap.add_argument("--predictions", required=True, help='JSONL of {"id": ..., "prediction": ...}')
    ap.add_argument("--tasks", default=None, help="comma-separated tasks, core6 or all (default: all)")
    ap.add_argument("--split", default="eval", choices=PUBLISHED_SPLITS)
    ap.add_argument("--difficulty", default=None, help="comma-separated levels (default: all)")
    ap.add_argument("--per-item", default=None, help="write per-item rows (JSONL) here")
    ap.add_argument("--out", default=None, help="write the summary (JSON) here")
    a = ap.parse_args(argv)
    from visro import dataset
    levels = a.difficulty.split(",") if a.difficulty else None
    predictions = load_predictions(a.predictions)
    records = list(dataset.iter_records(a.root, a.tasks, a.split, levels))
    rows = score_records(records, predictions)
    summary = summarize(rows)
    known = {r["id"] for r in records}
    summary.update(split=a.split, missing=sum(r["missing"] for r in rows),
                   unmatched_predictions=sum(1 for k in predictions if k not in known))
    if a.per_item:
        with open(a.per_item, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    width = max([len(t) for t in summary["tasks"]] + [5])
    for task, v in summary["tasks"].items():
        print(f"{task:<{width}}  {v['correct']:>6}/{v['n']:<6} {100 * v['accuracy']:6.2f}")
    print(f"{'macro':<{width}}  {'':>13} {100 * summary['macro_accuracy']:6.2f}")
    print(f"{'micro':<{width}}  {summary['correct']:>6}/{summary['n']:<6} {100 * summary['micro_accuracy']:6.2f}")
    if summary["missing"] or summary["unmatched_predictions"]:
        print(f"missing predictions: {summary['missing']} (scored wrong); predictions for other ids: "
              f"{summary['unmatched_predictions']}", file=sys.stderr)
    return summary


def cli():
    """Console entry point: main() returns a summary, not a process exit code."""
    main()


if __name__ == "__main__":
    cli()
