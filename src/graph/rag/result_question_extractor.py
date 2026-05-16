"""Extract explicit questions from retrieved RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_ID_KEYS = ("id", "unit_id", "source_id")
_QUESTION_RE = re.compile(r"[^?]+\?")
_QUESTION_WORDS = {"why", "how", "what", "when", "where", "who"}


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            if (text := _string(value)):
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _content(result: Any) -> str:
    return "\n".join(
        str(value)
        for key in _TEXT_KEYS
        for value in _candidate_values(result, key)
        if value is not _MISSING and value is not None and str(value).strip()
    )


def _question_type(question: str) -> str:
    match = re.match(r"^[\s\"'(*\-\d.)]*([A-Za-z]+)\b", question)
    if not match:
        return "other"
    word = match.group(1).casefold()
    return word if word in _QUESTION_WORDS else "other"


def _questions(text: str) -> Iterable[str]:
    for match in _QUESTION_RE.finditer(text):
        candidate = re.split(r"[.!]\s+", match.group(0))[-1]
        question = _string(candidate)
        if question is not None:
            yield question


def extract_result_questions(results: Iterable[Any], *, max_questions: int = 20) -> list[dict[str, Any]]:
    """Return question-mark terminated questions found in retrieved result text."""
    if not isinstance(max_questions, int) or isinstance(max_questions, bool) or max_questions < 0:
        raise ValueError("max_questions must be a non-negative integer")
    if max_questions == 0:
        return []

    rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(results):
        content = _content(result)
        if not content:
            continue
        result_id = _result_id(result, result_index)
        for question in _questions(content):
            rows.append(
                {
                    "result_id": result_id,
                    "question": question,
                    "question_type": _question_type(question),
                    "position": len(rows),
                }
            )
            if len(rows) >= max_questions:
                return rows
    return rows
