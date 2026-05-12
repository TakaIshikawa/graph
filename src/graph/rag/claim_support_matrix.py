"""Build a compact support matrix for intended RAG answer claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "snippet", "snippets")
_TAG_KEYS = ("tags",)
_KEYWORD_VALUE_KEYS = ("keyword", "term", "phrase", "key", "value", "tag")


def _validate_min_overlap(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_overlap must be a positive integer")
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


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


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


def _tokens(value: Any) -> set[str]:
    text = _string(value)
    if text is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    }


def _result_terms(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _TEXT_KEYS:
        for value in _iter_string_values(_result_value(result, key)):
            terms.update(_tokens(value))
    for key in _TAG_KEYS:
        for value in _iter_string_values(_result_value(result, key)):
            terms.update(_tokens(value))
    return terms


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source_project(result: Any) -> str:
    value = _string(_result_value(result, "source_project"))
    return value or "unknown"


def _support_level(support_count: int) -> str:
    if support_count == 0:
        return "none"
    if support_count == 1:
        return "weak"
    return "strong"


def build_claim_support_matrix(
    results: Iterable[Any],
    claims: Iterable[str],
    *,
    min_overlap: int = 1,
) -> list[dict[str, Any]]:
    """Match claim terms against retrieved result text and tags."""
    min_overlap_value = _validate_min_overlap(min_overlap)
    candidates = [
        {
            "result_id": _result_id(result, index),
            "source_project": _source_project(result),
            "terms": _result_terms(result),
        }
        for index, result in enumerate(results)
    ]

    rows = []
    for claim in claims:
        claim_text = _string(claim) or ""
        claim_terms = _tokens(claim_text)
        matches = []
        matched_terms: set[str] = set()

        for candidate in candidates:
            overlap = claim_terms & candidate["terms"]
            if len(overlap) >= min_overlap_value:
                matches.append(candidate)
                matched_terms.update(overlap)

        supporting_result_ids = sorted(candidate["result_id"] for candidate in matches)
        source_projects = sorted({candidate["source_project"] for candidate in matches})
        support_count = len(supporting_result_ids)
        rows.append(
            {
                "claim": claim_text,
                "supporting_result_ids": supporting_result_ids,
                "source_projects": source_projects,
                "matched_terms": sorted(matched_terms),
                "support_count": support_count,
                "support_level": _support_level(support_count),
            }
        )

    return rows
