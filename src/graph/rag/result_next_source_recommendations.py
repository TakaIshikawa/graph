"""Recommend source categories to retrieve next for RAG coverage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_date, source_id, string, value

_CATEGORIES = ("primary", "recent", "dissenting", "local", "methodology")


def recommend_result_next_sources(results: Iterable[Any], query_intent_hints: Iterable[str] | None = None) -> dict[str, Any]:
    """Return ranked missing source category recommendations."""
    records = list(results)
    hints = {str(hint).casefold() for hint in (query_intent_hints or [])}
    present = {category: False for category in _CATEGORIES}
    for record in records:
        text = " ".join([content_text(record), string(value(record, "source_type")) or "", source_id(record) or ""]).casefold()
        present["primary"] |= any(term in text for term in ("primary", "official", "original", "filing", "dataset"))
        present["recent"] |= result_date(record) is not None or any(term in text for term in ("recent", "latest", "current", "2025", "2026"))
        present["dissenting"] |= any(term in text for term in ("dissent", "contrary", "critique", "opposes", "however"))
        present["local"] |= any(term in text for term in ("local", "city", "county", "regional", "municipal"))
        present["methodology"] |= any(term in text for term in ("method", "methodology", "sample", "data", "survey"))

    recommendations = []
    for category in _CATEGORIES:
        if present[category]:
            continue
        priority = _priority(category, hints)
        recommendations.append(
            {
                "source_category": f"{category}_source",
                "rank": 0,
                "priority": priority,
                "reason_codes": [f"MISSING_{category.upper()}"],
            }
        )
    recommendations.sort(key=lambda row: (-row["priority"], row["source_category"]))
    for index, row in enumerate(recommendations, start=1):
        row["rank"] = index
    return {"recommendations": recommendations, "coverage": present}


def _priority(category: str, hints: set[str]) -> int:
    priority = {"primary": 90, "recent": 80, "dissenting": 70, "methodology": 60, "local": 50}[category]
    if category in hints or f"{category}_source" in hints:
        priority += 100
    return priority
