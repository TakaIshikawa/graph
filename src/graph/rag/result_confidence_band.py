"""Estimate downstream answer confidence from RAG result quality signals."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import any_present, coerce_now, result_date, source_id, value


def estimate_result_confidence_band(results: Iterable[Any], *, now: Any = None) -> dict[str, Any]:
    """Convert result quality signals into a low, medium, or high confidence band."""
    rows = list(results)
    if not rows:
        return {
            "confidence_band": "low",
            "confidence_score": 0.0,
            "contributing_factors": [],
            "downgrade_reasons": ["empty result set"],
        }

    today = coerce_now(now)
    factors: list[str] = []
    downgrades: list[str] = []
    score = 0.2

    count_score = min(len(rows) / 4, 1.0) * 0.25
    score += count_score
    factors.append(f"{len(rows)} evidence results")

    sources = {source_id(row) for row in rows if source_id(row)}
    if len(sources) >= 2:
        score += 0.2
        factors.append("diverse sources")
    else:
        downgrades.append("low source diversity")

    cited = sum(1 for row in rows if any_present(row, ("citation", "url", "source_url", "id")))
    if cited == len(rows):
        score += 0.2
        factors.append("citation metadata available")
    else:
        downgrades.append("missing citation metadata")

    dates = [parsed for row in rows if (parsed := result_date(row)) is not None]
    if dates and max((today - parsed).days for parsed in dates) <= 365:
        score += 0.15
        factors.append("freshness metadata is recent")
    else:
        downgrades.append("stale or missing freshness metadata")

    if any(value(row, "contradiction") is True or value(row, "contradiction_flag") is True for row in rows):
        score -= 0.25
        downgrades.append("contradiction flag present")

    score = round(max(0.0, min(1.0, score)), 2)
    band = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return {
        "confidence_band": band,
        "confidence_score": score,
        "contributing_factors": factors,
        "downgrade_reasons": downgrades,
    }
