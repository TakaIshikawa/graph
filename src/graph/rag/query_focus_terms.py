"""Extract deterministic focus terms from RAG queries and retrieved results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TAG_KEYS = ("tags",)
_KEYWORD_KEYS = ("keywords", "keyphrases", "metadata_keywords")
_KEYWORD_VALUE_KEYS = ("keyword", "term", "phrase", "key", "value", "tag")


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _result_value(result: Any, key: str) -> Any:
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
            return unit_metadata.get(key, _MISSING)

    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _source_key(result: Any, index: int) -> str:
    source_project = _string_value(_result_value(result, "source_project"))
    source_id = _string_value(_result_value(result, "source_id")) or _string_value(_result_value(result, "id"))
    if source_project and source_id:
        return f"{source_project}:{source_id}"
    if source_project:
        return source_project
    if source_id:
        return source_id
    return f"result-{index + 1}"


def _tokens(text: Any) -> list[str]:
    value = _string_value(text)
    if value is None:
        return []
    return [
        token
        for token in TOKEN_RE.findall(value.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    ]


def _iter_string_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _KEYWORD_VALUE_KEYS:
            item = _string_value(value.get(key, _MISSING))
            if item is not None:
                return [item]
        return [_string_value(key) for key in value if _string_value(key) is not None]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        strings: set[str] = set()
        for item in value:
            if isinstance(item, Mapping):
                strings.update(_iter_string_values(item))
            elif (string := _string_value(item)) is not None:
                strings.add(string)
        return sorted(strings)
    string = _string_value(value)
    return [] if string is None else [string]


def _result_terms(result: Any) -> set[str]:
    terms = set(_tokens(_result_value(result, "title")))
    for key in _TAG_KEYS + _KEYWORD_KEYS:
        for value in _iter_string_values(_result_value(result, key)):
            terms.update(_tokens(value))
    return terms


def extract_query_focus_terms(
    query: str,
    results: Iterable[Any] | None = None,
    *,
    max_terms: int = 12,
) -> list[dict[str, Any]]:
    """Return normalized focus terms from a query, optionally boosted by results."""
    max_terms_value = _validate_positive_int(max_terms, "max_terms")
    query_counts = Counter(_tokens(query))
    result_counts: dict[str, set[str]] = defaultdict(set)

    for index, result in enumerate(results or ()):
        source = _source_key(result, index)
        for term in _result_terms(result):
            if term in query_counts:
                result_counts[term].add(source)

    rows = []
    for term, query_count in query_counts.items():
        sources = sorted(result_counts.get(term, set()))
        score = round(query_count + len(sources) * 0.5, 3)
        rows.append(
            {
                "term": term,
                "score": score,
                "query_count": query_count,
                "result_count": len(sources),
                "sources": sources,
            }
        )

    rows.sort(key=lambda item: (-item["score"], -item["query_count"], -item["result_count"], item["term"]))
    return rows[:max_terms_value]
