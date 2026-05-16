"""Summarize numeric facts found in retrieved RAG/search evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_ID_KEYS = ("id", "unit_id", "source_id")
_TERM_RE = re.compile(r"[\w-]+")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
_CURRENCY_RE = re.compile(r"(?<!\w)(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP))\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s+[A-Z]|\s*$)|$)")


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
    return " ".join(
        text
        for key in _TEXT_KEYS
        for value in _candidate_values(result, key)
        if (text := _string(value)) is not None
    )


def _query_terms(query: Any) -> list[str]:
    if query is None:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for term in _TERM_RE.findall(str(query).casefold()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < used_end and end > used_start for used_start, used_end in occupied)


def _context(text: str, start: int, end: int, *, max_length: int = 140) -> str:
    for match in _SENTENCE_RE.finditer(text):
        if match.start() <= start and end <= match.end():
            context = _string(match.group(0)) or ""
            break
    else:
        window_start = max(0, start - max_length // 2)
        window_end = min(len(text), end + max_length // 2)
        context = _string(text[window_start:window_end]) or ""

    if len(context) <= max_length:
        return context
    offset = max(context.find(text[start:end]), 0)
    clip_start = max(0, offset - max_length // 2)
    clip_end = min(len(context), clip_start + max_length)
    clip_start = max(0, clip_end - max_length)
    return context[clip_start:clip_end].strip()


def _matched_query_terms(context: str, query_terms: list[str]) -> list[str]:
    context_terms = set(_TERM_RE.findall(context.casefold()))
    return [term for term in query_terms if term in context_terms]


def _matches(text: str) -> Iterable[tuple[int, int, str, str]]:
    occupied: list[tuple[int, int]] = []
    patterns = (
        ("date", _DATE_RE),
        ("currency", _CURRENCY_RE),
        ("percent", _PERCENT_RE),
        ("year", _YEAR_RE),
        ("number", _NUMBER_RE),
    )
    candidates: list[tuple[int, int, str, str]] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, occupied):
                continue
            occupied.append(span)
            candidates.append((span[0], span[1], kind, " ".join(match.group(0).split())))
    return sorted(candidates, key=lambda item: item[:2])


def summarize_evidence_numbers(
    results: Iterable[Any],
    query: Any = None,
    *,
    max_numbers: int = 20,
) -> list[dict[str, Any]]:
    """Return numeric evidence facts with compact surrounding context."""
    if not isinstance(max_numbers, int) or isinstance(max_numbers, bool) or max_numbers < 0:
        raise ValueError("max_numbers must be a non-negative integer")
    if max_numbers == 0:
        return []

    query_terms = _query_terms(query)
    rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(results):
        content = _content(result)
        if not content:
            continue
        result_id = _result_id(result, result_index)
        for start, end, kind, value in _matches(content):
            context = _context(content, start, end)
            rows.append(
                {
                    "result_id": result_id,
                    "value": value,
                    "kind": kind,
                    "context": context,
                    "matched_query_terms": _matched_query_terms(context, query_terms),
                    "position": len(rows),
                }
            )
            if len(rows) >= max_numbers:
                return rows
    return rows
