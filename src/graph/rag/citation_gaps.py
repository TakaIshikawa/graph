"""Prioritize retrieved RAG results that need citation support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()

_CITATION_SIGNAL_KEYS = (
    "citations",
    "references",
    "url",
    "source_url",
    "doi",
    "isbn",
    "cited_by",
)
_SOURCE_KEYS = ("source_project", "source", "source_name", "domain")


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


def _has_signal(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, Mapping):
        return any(_has_signal(nested) for nested in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_signal(nested) for nested in value)
    return _string_value(value) is not None


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        return [_string_value(key) or "" for key in sorted(value)]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return sorted({text for item in value if (text := _string_value(item)) is not None})
    text = _string_value(value)
    return [] if text is None else [text]


def _tokens(value: Any) -> set[str]:
    text = _string_value(value)
    if text is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    }


def _query_terms(query: str | None) -> set[str]:
    return _tokens(query)


def _result_terms(result: Any) -> set[str]:
    terms = set(_tokens(_result_value(result, "title")))
    terms.update(_tokens(_result_value(result, "content")))
    for tag in _iter_strings(_result_value(result, "tags")):
        terms.update(_tokens(tag))
    metadata = _result_value(result, "metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            terms.update(_tokens(key))
            if isinstance(value, str | int | float | bool):
                terms.update(_tokens(value))
            elif isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
                for item in value:
                    terms.update(_tokens(item))
    return terms


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source_project(result: Any) -> str | None:
    for key in _SOURCE_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return None


def _missing_reasons(matched_keys: list[str], query_matches: list[str]) -> list[str]:
    reasons: list[str] = []
    missing_keys = [key for key in _CITATION_SIGNAL_KEYS if key not in matched_keys]
    if missing_keys:
        reasons.append(f"missing citation signals: {', '.join(missing_keys)}")
    if not matched_keys:
        reasons.append("no citation metadata found")
    if query_matches:
        reasons.append(f"matches query terms: {', '.join(query_matches)}")
    return reasons


def prioritize_citation_gaps(
    results: Iterable[Any],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return ranked citation support priorities for retrieved results."""
    terms = _query_terms(query)
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        matched_keys = [key for key in _CITATION_SIGNAL_KEYS if _has_signal(_result_value(result, key))]
        query_matches = sorted(terms & _result_terms(result))
        missing_count = len(_CITATION_SIGNAL_KEYS) - len(matched_keys)
        priority_score = round(missing_count + len(query_matches) * 0.5, 3)

        rows.append(
            {
                "result_id": _result_id(result, index),
                "title": _string_value(_result_value(result, "title")),
                "source_project": _source_project(result),
                "citation_signal_count": len(matched_keys),
                "missing_signal_reasons": _missing_reasons(matched_keys, query_matches),
                "matched_query_terms": query_matches,
                "priority_score": priority_score,
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            int(item["citation_signal_count"]),
            str(item["source_project"] or "").casefold(),
            str(item["title"] or "").casefold(),
            str(item["result_id"]).casefold(),
        )
    )
    return rows
