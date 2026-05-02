"""Source agreement scoring for retrieved RAG/search results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_SOURCE_PROJECT_KEYS = ("source_project", "source", "project")
_KEYWORD_KEYS = ("keywords", "extracted_keywords", "keyphrases")
_KEYWORD_VALUE_KEYS = ("keyword", "phrase", "key", "value", "term", "tag")


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
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


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _result_source_project(result: Any) -> str:
    for key in _SOURCE_PROJECT_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return "unknown"


def _iter_string_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        values = _mapping_string_values(value)
        return sorted(item for item in values if item is not None)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        values = {
            string
            for item in value
            for string in (
                _mapping_string_values(item)
                if isinstance(item, Mapping)
                else [_string_value(item)]
            )
        }
        return sorted(item for item in values if item is not None)
    string = _string_value(value)
    return [] if string is None else [string]


def _mapping_string_values(value: Mapping[Any, Any]) -> list[str | None]:
    for key in _KEYWORD_VALUE_KEYS:
        string = _string_value(value.get(key, _MISSING))
        if string is not None:
            return [string]
    return [_string_value(key) for key in value]


def _normalize_phrase(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    tokens = TOKEN_RE.findall(text.casefold())
    if not tokens:
        return None
    return " ".join(tokens)


def _tokens_for_text(value: Any, *, min_term_length: int) -> set[str]:
    text = _string_value(value)
    if text is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) >= min_term_length and token not in COMMON_STOPWORDS
    }


def _tag_keys(result: Any) -> set[str]:
    return {
        key
        for tag in _iter_string_values(_result_value(result, "tags"))
        if (key := _normalize_phrase(tag)) is not None
    }


def _explicit_keyword_keys(result: Any) -> set[str]:
    keys: set[str] = set()
    for key in _KEYWORD_KEYS:
        for value in _iter_string_values(_result_value(result, key)):
            normalized = _normalize_phrase(value)
            if normalized is not None:
                keys.add(normalized)
    return keys


def _extracted_keyword_keys(result: Any, *, min_term_length: int) -> set[str]:
    keys = _explicit_keyword_keys(result)
    if keys:
        return keys

    title_terms = _tokens_for_text(
        _result_value(result, "title"),
        min_term_length=min_term_length,
    )
    content_terms = _tokens_for_text(
        _result_value(result, "content"),
        min_term_length=min_term_length,
    )
    tag_terms = set().union(
        *(
            _tokens_for_text(tag, min_term_length=min_term_length)
            for tag in _iter_string_values(_result_value(result, "tags"))
        )
    )

    return title_terms | tag_terms | content_terms


def _term_keys(result: Any, *, min_term_length: int) -> set[str]:
    return _tokens_for_text(
        " ".join(
            value
            for value in (
                _string_value(_result_value(result, "title")),
                _string_value(_result_value(result, "content")),
            )
            if value is not None
        ),
        min_term_length=min_term_length,
    )


def score_source_agreement(
    results: Iterable[Any],
    *,
    min_source_count: int = 1,
    limit: int | None = None,
    min_term_length: int = 3,
) -> list[dict[str, Any]]:
    """Return evidence keys ranked by agreement across distinct source projects."""
    min_source_count_value = _validate_positive_int(
        min_source_count,
        "min_source_count",
    )
    limit_value = _validate_non_negative_int(limit, "limit")
    min_term_length_value = _validate_positive_int(
        min_term_length,
        "min_term_length",
    )
    result_list = list(results)

    total_source_projects = {
        _result_source_project(result) for result in result_list
    }
    total_source_count = max(len(total_source_projects), 1)

    support: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"source_projects": set(), "unit_ids": set()}
    )
    for index, result in enumerate(result_list):
        unit_id = _result_id(result, index)
        source_project = _result_source_project(result)
        evidence_by_type = {
            "tag": _tag_keys(result),
            "keyword": _extracted_keyword_keys(
                result,
                min_term_length=min_term_length_value,
            ),
            "term": _term_keys(result, min_term_length=min_term_length_value),
        }
        for evidence_type, evidence_keys in evidence_by_type.items():
            for evidence_key in evidence_keys:
                bucket = support[(evidence_type, evidence_key)]
                bucket["source_projects"].add(source_project)
                bucket["unit_ids"].add(unit_id)

    rows = []
    for (evidence_type, evidence_key), bucket in support.items():
        source_projects = sorted(bucket["source_projects"])
        if len(source_projects) < min_source_count_value:
            continue
        unit_ids = sorted(bucket["unit_ids"])
        rows.append(
            {
                "evidence_type": evidence_type,
                "evidence_key": evidence_key,
                "supporting_source_projects": source_projects,
                "supporting_unit_ids": unit_ids,
                "source_count": len(source_projects),
                "unit_count": len(unit_ids),
                "agreement_score": round(len(source_projects) / total_source_count, 6),
            }
        )

    rows.sort(
        key=lambda item: (
            -item["agreement_score"],
            -item["source_count"],
            -item["unit_count"],
            item["evidence_type"],
            item["evidence_key"],
        )
    )
    if limit_value is not None:
        rows = rows[:limit_value]
    return rows
