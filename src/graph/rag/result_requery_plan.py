"""Build targeted follow-up retrieval plans for weak RAG results."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import any_present, coerce_now, result_date, source_id


def build_result_requery_plan(query: str, results: Iterable[Any], *, now: Any = None) -> dict[str, Any]:
    """Inspect results and propose deterministic follow-up queries."""
    rows = list(results)
    today = coerce_now(now)
    reasons: list[str] = []
    followups: list[str] = []

    if not rows:
        reasons.append("empty results")
        followups.append(f"{query} authoritative sources")

    citation_count = sum(1 for row in rows if any_present(row, ("citation", "url", "source_url", "id")))
    if rows and citation_count < len(rows):
        reasons.append("missing citation metadata")
        followups.append(f"{query} cited sources")

    dated = [parsed for row in rows if (parsed := result_date(row)) is not None]
    if rows and (not dated or all((today - parsed).days > 365 for parsed in dated)):
        reasons.append("stale or missing dates")
        followups.append(f"{query} latest updates")

    sources = {source_id(row) for row in rows if source_id(row)}
    if len(rows) >= 2 and len(sources) < 2:
        reasons.append("low source diversity")
        followups.append(f"{query} alternative sources")

    priority = "high" if not rows or len(reasons) >= 3 else "medium" if reasons else "low"
    return {
        "requery_needed": bool(reasons),
        "followup_queries": _unique(followups),
        "reasons": reasons,
        "priority": priority,
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(value.split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
