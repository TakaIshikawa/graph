"""Build deterministic explanation labels for retrieved RAG results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_CITATION_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
    "doi",
    "pmid",
    "arxiv_id",
    "isbn",
    "citation",
    "citations",
    "citation_count",
)
_DATE_KEYS = (
    "published_at",
    "publication_date",
    "updated_at",
    "created_at",
    "timestamp",
    "date",
)
_KEYWORD_VALUE_KEYS = ("keyword", "term", "phrase", "key", "value", "tag")


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _tuple_score(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return _MISSING


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

    if key == "score":
        return _tuple_score(result)
    return value


def _metadata(result: Any) -> Mapping[str, Any]:
    payload = _payload(result)
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        return metadata
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata
    return {}


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _float(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _tokens(value: Any) -> set[str]:
    text = _string(value)
    if text is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    }


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _KEYWORD_VALUE_KEYS:
            item = _string(value.get(key, _MISSING))
            if item is not None:
                return [item]
        strings: set[str] = set()
        for key, item in value.items():
            if (key_string := _string(key)) is not None:
                strings.add(key_string)
            strings.update(_iter_strings(item))
        return sorted(strings)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        strings: set[str] = set()
        for item in value:
            strings.update(_iter_strings(item))
        return sorted(strings)
    string = _string(value)
    return [] if string is None else [string]


def _terms_in(value: Any, query_terms: set[str]) -> set[str]:
    return _tokens(value) & query_terms


def _terms_in_values(values: Iterable[Any], query_terms: set[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        terms.update(_terms_in(value, query_terms))
    return terms


def _has_any(result: Any, keys: tuple[str, ...]) -> bool:
    return any(_string(_value(result, key)) is not None for key in keys)


def _summary(labels: list[str], matched_terms: list[str], max_reasons: int) -> str:
    if "weak_match" in labels:
        return "Weak query match; no normalized query terms were found."
    parts = []
    if matched_terms:
        parts.append("matched " + ", ".join(matched_terms))
    label_phrases = {
        "title_match": "title match",
        "content_match": "content match",
        "tag_match": "tag match",
        "metadata_match": "metadata match",
        "cited": "citation present",
        "dated": "date present",
        "high_score": "high retrieval score",
    }
    for label in labels:
        if label in label_phrases:
            parts.append(label_phrases[label])
    return "; ".join(parts[:max_reasons])


def explain_rag_results(
    results: Iterable[Any],
    query: str,
    *,
    max_reasons: int = 4,
) -> list[dict[str, Any]]:
    """Return explanation labels for each result in input order."""
    max_reasons_value = _validate_positive_int(max_reasons, "max_reasons")
    query_terms = _tokens(query)
    explanations: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        title_terms = _terms_in(_value(result, "title"), query_terms)
        content_terms = _terms_in_values(
            (_value(result, key) for key in ("content", "text", "summary", "snippet")),
            query_terms,
        )
        tag_terms = _terms_in_values(_iter_strings(_value(result, "tags")), query_terms)
        metadata_terms = _terms_in_values(_iter_strings(_metadata(result)), query_terms)
        matched_terms = sorted(title_terms | content_terms | tag_terms | metadata_terms)

        labels = []
        if title_terms:
            labels.append("title_match")
        if content_terms:
            labels.append("content_match")
        if tag_terms:
            labels.append("tag_match")
        if metadata_terms:
            labels.append("metadata_match")
        if _has_any(result, _CITATION_KEYS):
            labels.append("cited")
        if _has_any(result, _DATE_KEYS):
            labels.append("dated")
        if (_float(_value(result, "score")) or 0.0) >= 0.75:
            labels.append("high_score")
        if not matched_terms:
            labels.append("weak_match")

        explanations.append(
            {
                "result_id": _id(result, index),
                "labels": labels,
                "matched_terms": matched_terms,
                "evidence_summary": _summary(labels, matched_terms, max_reasons_value),
            }
        )

    return explanations
