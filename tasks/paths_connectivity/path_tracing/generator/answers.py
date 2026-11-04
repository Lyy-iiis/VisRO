"""Parse a Path Tracing answer text into the structured gold of its question type."""

from __future__ import annotations

import re
from typing import Any


def _clean(value: Any) -> str:
    """Strip a leading ``answer:``, quotes, backticks and ``\\boxed{}`` from an answer text."""
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`'\"")
    text = text.replace("\\boxed{", "").replace("}", "")
    return text.strip()


def _marker_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _clean(value).upper())


def _parse_boolean(value: Any) -> bool:
    tokens = re.findall(r"[a-z0-9]+", _clean(value).casefold())
    if tokens and tokens[0] in {"true", "yes", "y", "1"}:
        return True
    if tokens and tokens[0] in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"{value!r} is not a recognized path-tracing boolean")


def _normalize_sequence(sequence: Any) -> list[str]:
    if not isinstance(sequence, (list, tuple)):
        raise ValueError("qa.answer_sequence must be an array")
    normalized = [_marker_token(marker) for marker in sequence]
    if not normalized or any(not marker for marker in normalized):
        raise ValueError("qa.answer_sequence must contain marker labels")
    return normalized


def _parse_sequence(
    value: Any,
    expected_sequence: list[str] | None = None,
) -> list[str]:
    raw = _clean(value).upper()
    compact = _marker_token(raw)
    if expected_sequence is None:
        # without an explicit token sequence every label is one character
        return list(compact)

    expected = _normalize_sequence(expected_sequence)
    tokens = [
        _marker_token(token)
        for token in re.findall(r"[A-Z0-9]+", raw)
    ]
    if len(tokens) == 1 and compact == "".join(expected):
        # answer_text concatenates the labels; resolve that compact gold with the
        # record's explicit token sequence
        return expected
    return [token for token in tokens if token]


def _parse_pairs(value: Any) -> list[list[str]]:
    raw = _clean(value).upper()
    pairs: set[tuple[str, str]] = set()
    for start, end in re.findall(r"([A-Z]+)\s*[-:=>]+\s*([0-9]+)", raw):
        pairs.add((_marker_token(start), _marker_token(end)))
    if not pairs:
        # no separators: pair adjacent letter/number tokens, so "A 1, B 3" keeps
        # each pair's direction
        tokens = [
            _marker_token(token)
            for token in re.findall(r"[A-Z]+|[0-9]+", raw)
        ]
        for start, end in zip(tokens, tokens[1:]):
            if re.fullmatch(r"[A-Z]+", start) and re.fullmatch(r"[0-9]+", end):
                pairs.add((start, end))
    if not pairs:
        raise ValueError(f"{value!r} does not contain a connected marker pair")
    return [[start, end] for start, end in sorted(pairs)]


def structured_answer(
    qa_type: str,
    answer_text: Any,
    answer_sequence: list[str] | None = None,
) -> Any:
    """The canonical structured gold of a Path Tracing answer: one marker label (one_marker), the
    label sequence (ordered), the sorted [start, end] pairs (all_pairs) or a bool (same_line)."""
    if qa_type == "ordered" and answer_sequence is not None:
        expected = _normalize_sequence(answer_sequence)
        parsed = _parse_sequence(answer_text, expected)
        if parsed != expected:
            raise ValueError(
                f"ordered answer_text parses to {parsed!r}, but answer_sequence is "
                f"{expected!r}"
            )
        return expected
    if qa_type == "one_marker":
        marker = _marker_token(answer_text)
        if not marker:
            raise ValueError(f"{answer_text!r} does not contain a marker label")
        return marker
    if qa_type == "same_line":
        return _parse_boolean(answer_text)
    if qa_type == "ordered":
        sequence = _parse_sequence(answer_text)
        if not sequence:
            raise ValueError(f"{answer_text!r} does not contain an ordered marker sequence")
        return sequence
    if qa_type == "all_pairs":
        return _parse_pairs(answer_text)
    raise ValueError(f"unknown path-tracing answer type {qa_type!r}")
