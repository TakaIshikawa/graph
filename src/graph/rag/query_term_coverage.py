"""Score how well retrieved RAG results cover normalized query terms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "text", "snippet", "summary", "excerpt")
_TERM_KEYS = ("tags", "keywords", "keyphrases")
_TERM_VALUE_KEYS = ("tag", "keyword", "term", "phrase", "key", "value")
_IDENTIFIER_KEYS = ("id", "result_id", "unit_id", "source_id")


def score_query_term_coverage(query: str, results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return per-result query term coverage rows for result-like objects."""
    query_terms = _unique_terms(query)
    query_term_set = set(query_terms)
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        result_terms = _result_terms(result)
        matched_terms = [term for term in query_terms if term in result_terms]
        missing_terms = [term for term in query_terms if term not in result_terms]
        coverage_score = round(len(matched_terms) / len(query_terms), 3) if query_terms else 0.0
        rows.append(
            {
                "result_id": _result_id(result, index),
                "matched_terms": matched_terms,
                "missing_terms": missing_terms,
                "coverage_score": coverage_score,
            }
        )

    return rows


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


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            metadata_value = unit_metadata.get(key, _MISSING)
            if metadata_value is not _MISSING and metadata_value is not None:
                return metadata_value

    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _tokens(value: Any) -> list[str]:
    text = _string(value)
    if text is None:
        return []
    return [
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    ]


def _unique_terms(value: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _tokens(value):
        if token not in seen:
            terms.append(token)
            seen.add(token)
    return terms


def _iter_text_values(value: Any) -> Iterable[str]:
    if value is _MISSING or value is None:
        return
    if isinstance(value, Mapping):
        for key in _TERM_VALUE_KEYS:
            text = _string(value.get(key, _MISSING))
            if text is not None:
                yield text
                return
        for key, nested in value.items():
            text = _string(key)
            if text is not None:
                yield text
            yield from _iter_text_values(nested)
        return
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _iter_text_values(item)
        return
    text = _string(value)
    if text is not None:
        yield text


def _result_terms(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _TEXT_KEYS:
        terms.update(_tokens(_value(result, key)))
    for key in _TERM_KEYS:
        for value in _iter_text_values(_value(result, key)):
            terms.update(_tokens(value))
    return terms


def _result_id(result: Any, index: int) -> str:
    for key in _IDENTIFIER_KEYS:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return f"result-{index + 1}"
