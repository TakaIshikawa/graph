"""Suggest deterministic query expansion terms from retrieved RAG results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "text", "summary", "snippet")
_TAG_KEYS = ("tags",)
_KEYWORD_KEYS = ("keywords", "keyphrases", "metadata_keywords")
_KEYWORD_VALUE_KEYS = ("keyword", "term", "phrase", "key", "value", "tag")


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
            metadata_value = unit_metadata.get(key, _MISSING)
            if metadata_value is not _MISSING and metadata_value is not None:
                return metadata_value

    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _tokens(value: Any) -> list[str]:
    text = _string_value(value)
    if text is None:
        return []
    return [
        token
        for token in TOKEN_RE.findall(text.casefold())
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


def _result_tokens(result: Any) -> list[str]:
    tokens: list[str] = []
    for key in _TEXT_KEYS:
        tokens.extend(_tokens(_result_value(result, key)))
    for key in _TAG_KEYS + _KEYWORD_KEYS:
        for value in _iter_string_values(_result_value(result, key)):
            tokens.extend(_tokens(value))
    return tokens


def suggest_query_expansion_terms(
    results: Iterable[Any],
    query: str,
    *,
    max_terms: int = 10,
) -> dict[str, Any]:
    """Return ranked expansion terms from result titles, content, tags, and keywords."""
    max_terms_value = _validate_positive_int(max_terms, "max_terms")
    query_terms = sorted(set(_tokens(query)))
    query_term_set = set(query_terms)
    counts: Counter[str] = Counter()
    supporting_ids: dict[str, set[str]] = defaultdict(set)

    for index, result in enumerate(results):
        result_id = _result_id(result, index)
        for token in _result_tokens(result):
            if token in query_term_set:
                continue
            counts[token] += 1
            supporting_ids[token].add(result_id)

    rows = [
        {
            "term": term,
            "frequency": frequency,
            "result_count": len(supporting_ids[term]),
            "supporting_result_ids": sorted(supporting_ids[term]),
        }
        for term, frequency in counts.items()
    ]
    rows.sort(key=lambda item: (-item["frequency"], -item["result_count"], item["term"]))
    expansion_terms = rows[:max_terms_value]

    return {
        "query_terms": query_terms,
        "expansion_terms": expansion_terms,
        "supporting_result_ids": {
            item["term"]: item["supporting_result_ids"] for item in expansion_terms
        },
    }
