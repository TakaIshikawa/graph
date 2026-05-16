"""Rank retrieved results for context inclusion."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import (
    any_present,
    coerce_now,
    content_text,
    number,
    result_date,
    result_id,
    source_id,
    value,
)

_CITATION_KEYS = ("url", "source_url", "doi", "citation", "citations", "permalink")
_PROVENANCE_KEYS = ("source", "source_id", "source_project", "author", "published_at", "updated_at")


def _age_score(result_day: date | None, today: date) -> tuple[float, str | None]:
    if result_day is None:
        return 0.0, None
    age_days = max((today - result_day).days, 0)
    if age_days <= 90:
        return 1.0, "recent metadata"
    if age_days <= 365:
        return 0.7, "current metadata"
    if age_days <= 1095:
        return 0.35, "aging metadata"
    return 0.1, "stale metadata"


def plan_context_priority(
    results: Iterable[Any],
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Return ordered result priorities for context inclusion."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    source_seen: set[str] = set()
    today = coerce_now()
    scored: list[dict[str, Any]] = []

    for index, result in enumerate(rows):
        rid = result_id(result, index)
        relevance = number(value(result, "score"))
        if relevance is None:
            relevance = number(value(result, "relevance_score")) or 0.0
        relevance = min(max(relevance, 0.0), 1.0)

        reasons: list[str] = []
        score = relevance * 50
        if relevance:
            reasons.append(f"retrieval score {relevance:.2f}")

        if any_present(result, _CITATION_KEYS):
            score += 15
            reasons.append("citation present")
        if any_present(result, _PROVENANCE_KEYS):
            score += 10
            reasons.append("provenance present")

        recency_score, recency_reason = _age_score(result_date(result), today)
        if recency_reason:
            score += recency_score * 15
            reasons.append(recency_reason)

        content_length = len(content_text(result))
        if 120 <= content_length <= 3000:
            score += 5
            reasons.append("usable content length")
        elif content_length:
            score += 2
            reasons.append("limited content length")

        sid = source_id(result) or "unknown"
        if sid not in source_seen:
            score += 5
            reasons.append("adds source diversity")
            source_seen.add(sid)

        scored.append(
            {
                "result_id": rid,
                "priority_score": round(score, 2),
                "rank": 0,
                "reasons": reasons or ["no priority signals"],
                "_index": index,
                "_source": sid,
            }
        )

    scored.sort(key=lambda item: (-item["priority_score"], item["_index"], item["result_id"]))
    limited = scored if max_items is None else scored[: max(0, int(max_items))]
    items = []
    for rank, item in enumerate(limited, start=1):
        public = {
            "result_id": item["result_id"],
            "priority_score": item["priority_score"],
            "rank": rank,
            "reasons": item["reasons"],
        }
        items.append(public)

    return {
        "items": items,
        "counts": {
            "result_count": len(rows),
            "returned_count": len(items),
            "source_count": len({source_id(result) or "unknown" for result in rows}),
        },
    }
