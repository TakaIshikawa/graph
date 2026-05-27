"""Classify RAG query scope with transparent heuristics."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_CUES = {
    "comparative": ("versus", "vs", "compare", "comparison", "better", "best", "tradeoff"),
    "procedural": ("how to", "steps", "guide", "workflow", "implement", "set up"),
    "temporal": ("latest", "recent", "current", "today", "trend", "2024", "2025", "2026"),
    "exploratory": ("overview", "ideas", "options", "explore", "what are", "landscape"),
}
_STRATEGIES = {
    "narrow": "retrieve focused exact-match evidence",
    "comparative": "retrieve balanced evidence for each option and comparison axis",
    "exploratory": "retrieve diverse overview sources before narrowing",
    "temporal": "prioritize recent evidence and date-bearing sources",
    "procedural": "retrieve step-by-step guides and implementation examples",
}


def classify_query_scope(query: Any) -> dict[str, Any]:
    """Return a primary query scope, matched cues, confidence, and retrieval strategy."""
    text = string(query) or ""
    folded = text.casefold()
    matched = {scope: [cue for cue in cues if _matches(folded, cue)] for scope, cues in _CUES.items()}
    priority = ("comparative", "procedural", "temporal", "exploratory")
    primary = next((scope for scope in priority if matched[scope]), "narrow")
    cues = matched.get(primary, [])
    confidence = 0.85 if len(cues) >= 2 else 0.75 if cues else 0.55

    return {
        "primary_scope": primary,
        "matched_cues": cues,
        "confidence": confidence,
        "recommended_retrieval_strategy": _STRATEGIES[primary],
    }


def _matches(text: str, cue: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text))
