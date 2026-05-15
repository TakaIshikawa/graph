"""Select compact evidence spans from retrieved RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_ID_KEYS = ("id", "unit_id", "source_id")


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


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

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _text(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first_text(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _text(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first_text(result, _ID_KEYS) or f"result-{index + 1}"


def _query_terms(query: Any) -> list[str]:
    if query is None:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[\w-]+", str(query).casefold()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _matches(text: str, terms: list[str]) -> list[tuple[int, int, int, str]]:
    lowered = text.casefold()
    found: list[tuple[int, int, int, str]] = []
    for term_index, term in enumerate(terms):
        for match in re.finditer(re.escape(term), lowered):
            found.append((match.start(), match.end(), term_index, term))
    return sorted(found, key=lambda item: (item[0], item[2]))


def _span_bounds(matches: list[tuple[int, int, int, str]], text: str, window: int) -> tuple[int, int]:
    text_length = len(text)
    first_start = matches[0][0]
    last_end = matches[-1][1]
    center = (first_start + last_end) // 2
    start = max(0, center - window // 2)
    end = min(text_length, start + window)
    start = max(0, end - window)
    if start > 0:
        next_space = text.find(" ", start, end)
        if next_space > start:
            start = next_space + 1
    if end < text_length:
        previous_space = text.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space
    return start, end


def _matched_terms(
    matches: list[tuple[int, int, int, str]],
    start: int,
    end: int,
) -> list[str]:
    by_term: dict[str, int] = {}
    for match_start, match_end, term_index, term in matches:
        if match_start >= start and match_end <= end:
            by_term.setdefault(term, term_index)
    return [term for term, _index in sorted(by_term.items(), key=lambda item: item[1])]


def select_evidence_spans(
    results: Iterable[Any],
    query: Any,
    *,
    window: int = 160,
    max_spans: int = 5,
) -> list[dict[str, Any]]:
    """Return compact snippets around case-insensitive query term matches."""
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(max_spans, int) or isinstance(max_spans, bool) or max_spans < 0:
        raise ValueError("max_spans must be a non-negative integer")

    terms = _query_terms(query)
    if not terms or max_spans == 0:
        return []

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for index, result in enumerate(results):
        content = _first_text(result, _TEXT_KEYS)
        if content is None:
            continue
        matches = _matches(content, terms)
        if not matches:
            continue
        start, end = _span_bounds(matches, content, window)
        span = content[start:end].strip()
        matched_terms = _matched_terms(matches, start, end)
        density = len([match for match in matches if start <= match[0] and match[1] <= end]) / max(
            len(span),
            1,
        )
        candidates.append(
            (
                -density,
                index,
                start,
                {
                    "result_id": _result_id(result, index),
                    "span": span,
                    "matched_terms": matched_terms,
                    "start": start,
                    "end": end,
                },
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [candidate[-1] for candidate in candidates[:max_spans]]
