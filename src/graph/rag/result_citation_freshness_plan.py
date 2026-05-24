"""Plan citation freshness checks for dated RAG results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date, result_id, source_id

_RECENCY_RE = re.compile(r"\b(?:latest|current|today|recent|newest|now|this year|up to date|updated)\b", re.I)


def plan_result_citation_freshness(query: str, results: Iterable[Any], reference_date: date | None = None) -> dict[str, Any]:
    """Classify result freshness and recommend citation refresh actions."""
    today = coerce_now(reference_date)
    strict = bool(_RECENCY_RE.search(str(query or "")))
    fresh_days, stale_days = (30, 180) if strict else (180, 1095)
    rows = []
    for index, result in enumerate(results):
        parsed = result_date(result)
        if parsed is None:
            status = "undated"
            age_days = None
            action = "check_or_replace"
        else:
            age_days = max((today - parsed).days, 0)
            if age_days <= fresh_days:
                status = "fresh"
                action = "keep"
            elif age_days <= stale_days:
                status = "aging"
                action = "check_freshness" if strict else "keep"
            else:
                status = "stale"
                action = "replace_or_update"
        rows.append(
            {
                "result_id": result_id(result, index),
                "source_id": source_id(result),
                "date": parsed.isoformat() if parsed else None,
                "age_days": age_days,
                "freshness_status": status,
                "recommended_action": action,
            }
        )
    return {"rows": rows, "refresh_needed_count": sum(1 for row in rows if row["recommended_action"] != "keep")}
