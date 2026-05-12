"""Flag RAG results that drift away from query focus terms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "snippet", "summary", "content", "text")
_TAG_KEYS = ("tags",)
_METADATA_KEYS = ("metadata",)
_METADATA_TEXT_KEYS = ("keywords", "keyphrases", "topic", "topics", "category", "categories", "source")
_KEYWORD_VALUE_KEYS = ("keyword", "term", "phrase", "key", "value", "tag", "name")


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


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


def _focus_terms(query: str) -> list[str]:
    counts = Counter(_tokens(query))
    return sorted(counts)


def _iter_string_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _KEYWORD_VALUE_KEYS:
            item = _string(value.get(key, _MISSING))
            if item is not None:
                return [item]
        return [_string(key) for key in value if _string(key) is not None]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        strings: set[str] = set()
        for item in value:
            if isinstance(item, Mapping):
                strings.update(_iter_string_values(item))
            elif (string := _string(item)) is not None:
                strings.add(string)
        return sorted(strings)
    string = _string(value)
    return [] if string is None else [string]


def _metadata_terms(metadata: Any) -> set[str]:
    terms: set[str] = set()
    if not isinstance(metadata, Mapping):
        return terms
    for key, value in metadata.items():
        if key in _METADATA_TEXT_KEYS:
            for item in _iter_string_values(value):
                terms.update(_tokens(item))
        elif isinstance(value, str | int | float | bool):
            terms.update(_tokens(value))
        elif isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
            for item in value:
                terms.update(_tokens(item))
    return terms


def _result_terms(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _TEXT_KEYS:
        terms.update(_tokens(_value(result, key)))
    for key in _TAG_KEYS:
        for item in _iter_string_values(_value(result, key)):
            terms.update(_tokens(item))
    for key in _METADATA_KEYS:
        terms.update(_metadata_terms(_value(result, key)))
    return terms


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "result_id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _title(result: Any) -> str | None:
    return _string(_value(result, "title"))


def analyze_query_drift(query: str, results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return per-result drift records by matching normalized query focus terms."""
    focus_terms = _focus_terms(query)
    focus_set = set(focus_terms)
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        result_terms = _result_terms(result)
        matched_terms = sorted(focus_set & result_terms)
        missing_terms = sorted(focus_set - result_terms)
        if focus_terms:
            drift_score = round(len(missing_terms) / len(focus_terms), 3)
        else:
            drift_score = 0.0
        rows.append(
            {
                "result_id": _result_id(result, index),
                "title": _title(result),
                "matched_terms": matched_terms,
                "missing_terms": missing_terms,
                "drift_score": drift_score,
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item["drift_score"]),
            len(item["matched_terms"]),
            str(item["title"] or "").casefold(),
            str(item["result_id"]).casefold(),
        )
    )
    return rows
