"""Detect comparison intents in RAG queries."""

from __future__ import annotations

from typing import Any

from graph.rag.query_comparison_requirement import detect_query_comparison_requirement


def detect_query_comparison_intent(query: str) -> list[dict[str, Any]]:
    result = detect_query_comparison_requirement(query)
    if not result["requires_comparison"]:
        return []
    return [
        {
            "comparison_type": result["comparison_type"],
            "matched_phrases": result["matched_terms"],
            "candidate_entities": result["entities"],
            "severity": "medium",
        }
    ]
