"""Plan deterministic source coverage strategy for RAG queries."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_SOURCE_TYPE_KEYS = ("source_type", "source_entity_type", "entity_type", "type", "kind")
_SOURCE_CUES: dict[str, tuple[str, ...]] = {
    "paper": (
        "paper",
        "papers",
        "publication",
        "publications",
        "study",
        "studies",
        "journal",
        "journals",
        "article",
        "articles",
    ),
    "doc": (
        "doc",
        "docs",
        "documentation",
        "manual",
        "manuals",
        "guide",
        "guides",
        "readme",
        "reference",
    ),
    "issue": (
        "issue",
        "issues",
        "ticket",
        "tickets",
        "bug",
        "bugs",
        "incident",
        "incidents",
    ),
    "note": (
        "note",
        "notes",
        "memo",
        "memos",
        "journal",
        "journals",
        "highlight",
        "highlights",
    ),
    "bookmark": (
        "bookmark",
        "bookmarks",
        "saved",
        "pocket",
        "pinboard",
        "link",
        "links",
    ),
    "citation": (
        "citation",
        "citations",
        "cite",
        "cites",
        "cited",
        "reference",
        "references",
        "bibliography",
        "bibliographic",
    ),
}
_SOURCE_ALIASES = {
    "papers": "paper",
    "publication": "paper",
    "publications": "paper",
    "study": "paper",
    "studies": "paper",
    "journal": "paper",
    "journals": "paper",
    "article": "paper",
    "articles": "paper",
    "docs": "doc",
    "documentation": "doc",
    "manual": "doc",
    "manuals": "doc",
    "guide": "doc",
    "guides": "doc",
    "readme": "doc",
    "reference_doc": "doc",
    "github_issue": "issue",
    "issues": "issue",
    "ticket": "issue",
    "tickets": "issue",
    "bug": "issue",
    "bugs": "issue",
    "notes": "note",
    "memo": "note",
    "memos": "note",
    "highlight": "note",
    "highlights": "note",
    "saved_item": "bookmark",
    "saved": "bookmark",
    "link": "bookmark",
    "links": "bookmark",
    "citations": "citation",
    "cite": "citation",
    "cites": "citation",
    "cited": "citation",
    "reference": "citation",
    "references": "citation",
    "bibliography": "citation",
    "bibliographic": "citation",
}


def plan_query_source_strategy(
    query: str,
    results: Iterable[Any] = (),
    required_source_types: Iterable[str] | None = None,
    min_source_types: int = 2,
) -> dict[str, Any]:
    """Return requested, observed, and missing source type coverage for a query."""
    normalized_query = _validate_query(query)
    minimum = _validate_min_source_types(min_source_types)
    result_list = list(results)

    requested = _requested_source_types(normalized_query)
    for source_type in _validate_required_source_types(required_source_types):
        if source_type not in requested:
            requested.append(source_type)

    observed = sorted(
        {
            source_type
            for result in result_list
            if (source_type := _source_type(result)) is not None
        }
    )
    source_type_count = len(observed)
    missing = sorted(source_type for source_type in requested if source_type not in observed)
    if source_type_count < minimum and "additional_source_type" not in missing:
        missing.append("additional_source_type")

    return {
        "query_terms": _query_terms(normalized_query),
        "requested_source_types": requested,
        "observed_source_types": observed,
        "missing_source_types": missing,
        "source_type_count": source_type_count,
        "needs_more_sources": bool(missing) or source_type_count < minimum,
        "recommendations": _recommendations(requested, observed, missing, minimum),
    }


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.strip().casefold().split())
    if not normalized:
        raise ValueError("query must be a non-empty string")
    return normalized


def _validate_min_source_types(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_source_types must be a positive integer")
    return value


def _validate_required_source_types(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    output: list[str] = []
    for value in values:
        normalized = _normalize_source_type(value)
        if normalized is None:
            raise ValueError("required_source_types must contain non-empty strings")
        if normalized not in output:
            output.append(normalized)
    return output


def _query_terms(query: str) -> list[str]:
    terms = [
        token
        for token in TOKEN_RE.findall(query)
        if len(token) > 1 and token not in COMMON_STOPWORDS
    ]
    return _dedupe(terms)


def _requested_source_types(query: str) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]*", query))
    requested: list[str] = []
    for source_type, cues in _SOURCE_CUES.items():
        if any(cue in tokens for cue in cues) and source_type not in requested:
            requested.append(source_type)
    return requested


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


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


def _normalize_source_type(value: Any) -> str | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if hasattr(value, "value"):
        value = value.value
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    if not text:
        return None
    if text in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[text]
    for alias, normalized in _SOURCE_ALIASES.items():
        if alias in text.split("_"):
            return normalized
    return text


def _source_type(result: Any) -> str | None:
    for key in _SOURCE_TYPE_KEYS:
        for value in _candidate_values(result, key):
            normalized = _normalize_source_type(value)
            if normalized is not None:
                return normalized
    return None


def _recommendations(
    requested: list[str],
    observed: list[str],
    missing: list[str],
    minimum: int,
) -> list[str]:
    recommendations: list[str] = []
    requested_missing = [source_type for source_type in missing if source_type != "additional_source_type"]
    if requested_missing:
        recommendations.append(
            "Retrieve requested source types: " + ", ".join(requested_missing) + "."
        )
    if not requested:
        recommendations.append("Add explicit source-type constraints if the answer needs provenance diversity.")
    if len(observed) < minimum:
        recommendations.append(
            f"Add at least {minimum - len(observed)} more distinct source type"
            f"{'' if minimum - len(observed) == 1 else 's'}."
        )
    if not recommendations:
        recommendations.append("Current results satisfy requested source-type coverage.")
    return recommendations
