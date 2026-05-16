"""Detect likely blind spots in retrieved RAG evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import (
    any_present,
    coerce_now,
    content_text,
    ordered_terms,
    result_date,
    source_id,
    string,
    tokens,
    value,
)

_URL_KEYS = ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri")
_PRIMARY_KEYS = ("is_primary_source", "primary_source", "source_type")


def _add(
    blind_spots: list[dict[str, str]],
    code: str,
    severity: str,
    reason: str,
    suggested_action: str,
) -> None:
    blind_spots.append(
        {
            "code": code,
            "severity": severity,
            "reason": reason,
            "suggested_action": suggested_action,
        }
    )


def _entity_type(result: Any) -> str:
    for key in ("entity_type", "type", "unit_type", "content_type"):
        text = string(value(result, key))
        if text is not None:
            return text
    return "unknown"


def _has_primary_or_url(result: Any) -> bool:
    if any_present(result, _URL_KEYS):
        return True
    source_type = (string(value(result, "source_type")) or "").casefold()
    if source_type in {"primary", "official"}:
        return True
    return any_present(result, _PRIMARY_KEYS)


def detect_result_blind_spots(query: Any, results: Iterable[Any]) -> dict[str, Any]:
    """Return blind spots suggested by query terms and result metadata."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    terms = ordered_terms(query)
    blind_spots: list[dict[str, str]] = []
    today = coerce_now()

    if not rows:
        _add(
            blind_spots,
            "no_results",
            "high",
            "No retrieved results are available for the query.",
            "Retrieve evidence before drafting an answer.",
        )
        return {
            "blind_spots": blind_spots,
            "counts": {"result_count": 0, "blind_spot_count": 1},
            "query_terms": terms,
        }

    dates = [result_date(result) for result in rows]
    if not any(day is not None and (today - day).days <= 365 for day in dates):
        _add(
            blind_spots,
            "missing_recent_evidence",
            "medium",
            "No result has date metadata from the last year.",
            "Add or verify recent evidence before making time-sensitive claims.",
        )

    entity_types = {_entity_type(result) for result in rows}
    if len(entity_types) == 1 and len(rows) > 1:
        _add(
            blind_spots,
            "single_entity_type",
            "medium",
            "Retrieved evidence covers only one entity type.",
            "Search for complementary entity types or formats.",
        )

    sources = {source_id(result) or "unknown" for result in rows}
    if len(sources) <= 1 and len(rows) > 1:
        _add(
            blind_spots,
            "narrow_source_set",
            "high",
            "Retrieved evidence comes from one source bucket.",
            "Add independent sources to test source diversity.",
        )

    if not any(_has_primary_or_url(result) for result in rows):
        _add(
            blind_spots,
            "no_primary_or_url_evidence",
            "high",
            "No result exposes primary-source or URL evidence.",
            "Prefer retrievable source URLs or primary-source records.",
        )

    if terms:
        covered = set()
        for result in rows:
            covered.update(tokens(content_text(result)) & set(terms))
        coverage = len(covered) / len(terms)
        if coverage < 0.5:
            _add(
                blind_spots,
                "weak_query_term_coverage",
                "medium",
                "Less than half of normalized query terms appear in retrieved evidence.",
                "Refine retrieval or broaden result inspection for missing query concepts.",
            )

    return {
        "blind_spots": blind_spots,
        "counts": {"result_count": len(rows), "blind_spot_count": len(blind_spots)},
        "query_terms": terms,
    }
