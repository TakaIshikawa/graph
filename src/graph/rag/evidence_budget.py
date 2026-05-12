"""Allocate a compact deterministic evidence budget from RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "text", "summary", "snippet")
_TAG_KEYS = ("tags",)
_KEYWORD_KEYS = ("keywords", "keyphrases", "metadata_keywords")
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


def _tokens(value: Any) -> list[str]:
    text = _string(value)
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


def _result_terms(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _TEXT_KEYS:
        terms.update(_tokens(_value(result, key)))
    for key in _TAG_KEYS + _KEYWORD_KEYS:
        for value in _iter_string_values(_value(result, key)):
            terms.update(_tokens(value))
    return terms


def _has_any(result: Any, keys: tuple[str, ...]) -> bool:
    return any(_string(_value(result, key)) is not None for key in keys)


def _source_project(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _candidate(result: Any, index: int, query_terms: set[str]) -> dict[str, Any]:
    terms = _result_terms(result)
    score = _float(_value(result, "score"))
    return {
        "rank": index + 1,
        "result_id": _id(result, index),
        "title": _string(_value(result, "title")),
        "source_project": _source_project(result),
        "score": score,
        "covered_terms": sorted(terms & query_terms),
        "has_citation": _has_any(result, _CITATION_KEYS),
        "has_date": _has_any(result, _DATE_KEYS),
        "_raw": result,
    }


def _score_candidate(
    candidate: dict[str, Any],
    *,
    covered_terms: set[str],
    source_counts: Counter[str],
    query_term_count: int,
    min_sources: int,
) -> tuple[float, float, int, int, str, int]:
    raw_score = candidate["score"] if candidate["score"] is not None else 0.0
    score_component = max(min(raw_score, 1.0), 0.0)
    new_terms = set(candidate["covered_terms"]) - covered_terms
    coverage_component = len(new_terms) / max(query_term_count, 1)
    source = candidate["source_project"]
    source_component = 1.0 if source_counts[source] == 0 and len(source_counts) < min_sources else 0.0
    citation_component = 1.0 if candidate["has_citation"] else 0.0
    date_component = 1.0 if candidate["has_date"] else 0.0
    utility = (
        score_component * 0.45
        + coverage_component * 0.3
        + source_component * 0.15
        + citation_component * 0.06
        + date_component * 0.04
    )
    return (
        utility,
        score_component,
        len(new_terms),
        len(candidate["covered_terms"]),
        candidate["result_id"],
        -candidate["rank"],
    )


def allocate_evidence_budget(
    results: Iterable[Any],
    query: str,
    *,
    max_results: int = 8,
    min_sources: int = 2,
) -> dict[str, Any]:
    """Select a compact result subset for answer context."""
    max_results_value = _validate_positive_int(max_results, "max_results")
    min_sources_value = _validate_positive_int(min_sources, "min_sources")
    query_terms = set(_tokens(query))
    candidates = [_candidate(result, index, query_terms) for index, result in enumerate(results)]
    remaining = candidates[:]
    selected: list[dict[str, Any]] = []
    covered_terms: set[str] = set()
    source_counts: Counter[str] = Counter()

    while remaining and len(selected) < max_results_value:
        best = max(
            remaining,
            key=lambda item: _score_candidate(
                item,
                covered_terms=covered_terms,
                source_counts=source_counts,
                query_term_count=len(query_terms),
                min_sources=min_sources_value,
            ),
        )
        remaining.remove(best)
        selected.append(best)
        covered_terms.update(best["covered_terms"])
        source_counts[best["source_project"]] += 1

    selected_ids = {item["result_id"] for item in selected}
    selected_results = [{key: value for key, value in item.items() if key != "_raw"} for item in selected]
    omitted_result_ids = [item["result_id"] for item in candidates if item["result_id"] not in selected_ids]

    return {
        "selected_results": selected_results,
        "omitted_result_ids": omitted_result_ids,
        "covered_terms": sorted(covered_terms),
        "missing_terms": sorted(query_terms - covered_terms),
        "source_counts": dict(sorted(source_counts.items())),
        "stats": {
            "total_results": len(candidates),
            "selected_count": len(selected_results),
            "omitted_count": len(omitted_result_ids),
            "max_results": max_results_value,
            "min_sources": min_sources_value,
            "query_term_count": len(query_terms),
        },
    }
